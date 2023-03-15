import functools
import time
import numpy as np
import torch
import flatdict

from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
from internal import camera_utils
from internal import sample_utils

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.

class RefNeRFSystem(LightningModule):
    def __init__(self, hparams, config, summary_writer):
        super(RefNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.config = config
        self.summary_writer = summary_writer
        dummy_rays = utils.dummy_rays()
        self.model = models.construct_model(dummy_rays, config)
        self.render_eval_fn = train_utils.create_render_fn(self.model)
        self.metric_harness = image.MetricHarness()
        self.reset_stats = True
        self.total_time = 0
        self.total_steps = 0

    def setup(self, stage):
        self.train_dataset = datasets.load_dataset('train', self.config.data_dir, self.config)
        self.val_dataset = datasets.load_dataset('test', self.config.data_dir, self.config)

    def configure_optimizers(self):
        self.optimizer, self.scheduler = train_utils.create_optimizer(self.config, self.model.parameters())
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          batch_size=1,
                          pin_memory=True,
                          collate_fn=lambda x : x[0])

    def val_dataloader(self):
        # must give 1 worker
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          batch_size=1,
                          pin_memory=True,
                          collate_fn=lambda x : x[0])
    
    def train_frac(self):
        return np.clip((self.global_step - 1) / (self.config.max_steps - 1), 0, 1)

    def training_step(self, batch, batch_idx):
        # clear stats for this iteration
        if self.reset_stats:
            self.stats_buffer = []
            self.train_start_time = time.time()
            self.reset_stats = False

        rays = batch.rays
        if self.config.cast_rays_in_train_step:
            rays = camera_utils.cast_ray_batch(
                self.train_dataset.cameras, rays, self.train_dataset.camtype, xnp=torch)

        renderings, ray_history = self.model(
            rays,
            train_frac=self.train_frac(),
            compute_extras=\
                self.config.compute_disp_metrics or 
                self.config.compute_normal_metrics or 
                self.config.sample_noise_size > 0)

        # calculate warmup ratio
        if self.config.consistency_warmup_steps > 0.:
            warmup_ratio = min(1., self.global_step/(self.config.consistency_warmup_steps*self.config.max_steps))
        else:
            warmup_ratio = 1.

        if self.config.sample_noise_size > 0 and (
           self.config.consistency_diffuse_coarse_loss_mult > 0 or
           self.config.consistency_specular_coarse_loss_mult > 0 or 
           self.config.consistency_normal_coarse_loss_mult > 0 or
           self.config.consistency_diffuse_loss_mult > 0 or
           self.config.consistency_specular_loss_mult > 0 or 
           self.config.consistency_normal_loss_mult > 0):
            noisy_rays = sample_utils.sample_noisy_rays(
                rays, renderings[-1], self.config.sample_angle_range, 
                self.config.sample_noise_size, self.config.sample_noise_angles, warmup_ratio)
            renderings_noise, ray_history_noise = self.model(
                noisy_rays,
                train_frac=self.train_frac(),
                compute_extras=True)

        losses = {}

        # calculate photometric error
        data_loss, self.stats = train_utils.compute_data_loss(batch, renderings, rays, self.config)
        losses['data'] = data_loss

        # calculate interlevel loss
        if self.config.interlevel_loss_mult > 0:
            losses['interlevel'] = train_utils.interlevel_loss(ray_history, self.config)

        # calculate normals orientation loss
        if (self.config.orientation_coarse_loss_mult > 0 or
                self.config.orientation_loss_mult > 0):
            losses['orientation'] = train_utils.orientation_loss(
                rays, self.model, ray_history, self.config)

        # calculate predicted normal loss
        if (self.config.predicted_normal_coarse_loss_mult > 0 or
                self.config.predicted_normal_loss_mult > 0):
            losses['predicted_normals'] = train_utils.predicted_normal_loss(
                self.model, ray_history, self.config)

        # calculate predicted consistency loss
        if self.config.sample_noise_size > 0 and (
           self.config.consistency_diffuse_coarse_loss_mult > 0 or
           self.config.consistency_specular_coarse_loss_mult > 0 or 
           self.config.consistency_normal_coarse_loss_mult > 0 or
           self.config.consistency_diffuse_loss_mult > 0 or
           self.config.consistency_specular_loss_mult > 0 or 
           self.config.consistency_normal_loss_mult > 0):
            consistency_losses = train_utils.noisy_consistency_loss(
                self.model, renderings, renderings_noise, self.config, warmup_ratio)
            (losses['diffuse_consistency'], 
             losses['specular_consistency'], 
             losses['normals_consistency']) = consistency_losses

        # calculate total loss
        loss = torch.sum(torch.stack(list(losses.values())))
        self.stats['loss'] = loss.detach().cpu()
        self.stats['losses'] = {key: losses[key].detach().cpu() for key in losses}


        # Calculate PSNR metric
        self.stats['psnrs'] = image.mse_to_psnr(self.stats['mses'])
        self.stats['psnr'] = self.stats['psnrs'][-1]

        self.log('train/loss', self.stats['loss'], prog_bar=True)
        self.log('train/psnr', self.stats['psnr'], prog_bar=True)

        return loss

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # Clip gradients
        if self.config.grad_max_val > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.config.grad_max_val)
        if self.config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_max_norm)

    def on_after_backward(self):
        # calculate average grad and stats
        params = dict(self.model.named_parameters())
        self.stats['weights_l2s'] = {k.replace('.', '/') : params[k].detach().norm() ** 2 for k in params}
        self.stats['grad_norms'] = {k.replace('.', '/') : params[k].grad.detach().cpu().norm() for k in params}
        self.stats['grad_maxes'] = {k.replace('.', '/') : params[k].grad.detach().cpu().abs().max() for k in params}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            # Log training summaries
            self.stats_buffer.append(self.stats)
            if self.global_step == 0 or self.global_step % self.config.print_every == 0:

                elapsed_time = time.time() - self.train_start_time
                steps_per_sec = self.config.print_every / elapsed_time
                rays_per_sec = self.config.batch_size * steps_per_sec

                # A robust approximation of total training time, in case of pre-emption.
                self.total_time += int(round(TIME_PRECISION * elapsed_time))
                self.total_steps += self.config.print_every
                approx_total_time = int(round(self.global_step * self.total_time / self.total_steps))

                # Stack stats_buffer along axis 0.
                fs = [dict(flatdict.FlatDict(s, delimiter='/')) for s in self.stats_buffer]
                stats_stacked = {k: torch.stack([f[k] for f in fs]) for k in fs[0].keys()}

                # Split every statistic that isn't a vector into a set of statistics.
                stats_split = {}
                for k, v in stats_stacked.items():
                    if v.ndim not in [1, 2] and v.shape[0] != len(self.stats_buffer):
                        raise ValueError('statistics must be of size [n], or [n, k].')
                    if v.ndim == 1:
                        stats_split[k] = v
                    elif v.ndim == 2:
                        for i, vi in enumerate(tuple(v.T)):
                            stats_split[f'{k}/{i}'] = vi

                # Summarize the entire histogram of each statistic.
                for k, v in stats_split.items():
                    self.summary_writer.add_histogram('train/' + k, v, self.global_step)

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: torch.mean(v) for k, v in stats_split.items()}
                max_stats = {k: torch.max(v) for k, v in stats_split.items()}

                # Summarize the mean and max of each statistic.
                for k, v in avg_stats.items():
                    self.summary_writer.add_scalar(f'train/avg_{k}', v, self.global_step)
                for k, v in max_stats.items():
                    self.summary_writer.add_scalar(f'train/max_{k}', v, self.global_step)
                self.summary_writer.add_scalar('train/num_params', sum(p.numel() for p in self.model.parameters()), self.global_step)
                self.summary_writer.add_scalar('train/learning_rate', *self.scheduler.get_last_lr(), self.global_step)

                self.summary_writer.add_scalar('train/steps_per_sec', steps_per_sec, self.global_step)
                self.summary_writer.add_scalar('train/rays_per_sec', rays_per_sec, self.global_step)
                self.summary_writer.add_scalar('train/avg_psnr_timed', avg_stats['psnr'],
                                    self.total_time // TIME_PRECISION)
                self.summary_writer.add_scalar('train/avg_psnr_timed_approx', avg_stats['psnr'],
                                    approx_total_time // TIME_PRECISION)

                # Reset everything we are tracking between summarizations.
                self.reset_stats = True

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            eval_start_time = time.time()
            (batch) = batch
            rgb, rays, normals = batch.rgb, batch.rays, batch.normals

            # Render test image.
            rendering = models.render_image(
                functools.partial(self.render_eval_fn, self.train_frac()),
                rays, self.config)

            # Log eval summaries.
            eval_time = time.time() - eval_start_time
            num_rays = np.prod(np.array(rays.directions.shape[:-1]))
            rays_per_sec = num_rays / eval_time
            self.summary_writer.add_scalar('val/rays_per_sec', rays_per_sec, self.global_step)
            
            log = {}

            # Compute metrics.
            metric = self.metric_harness(rendering['rgb'], torch.tensor(rgb, device=rendering['rgb'].device))
            psnr = metric['psnr']
            if np.isnan(psnr):
                psnr = 0.

            log = {'psnr': psnr}

            # Log images to tensorboard.
            vis_suite = vis.visualize_suite(rendering, rays)
            self.summary_writer.add_image(
                'val/true_color', rgb, self.global_step, dataformats='HWC')
            if normals is not None:
                self.summary_writer.add_image(
                    'val/true_normals', normals / 2. + 0.5, self.global_step,
                    dataformats='HWC')
            for k, v in vis_suite.items():
                self.summary_writer.add_image(
                    'val/output_' + k, v, self.global_step,
                    dataformats='HWC' if len(v.shape) == 3 else 'HW')
            for k, v in log.items():
                self.summary_writer.add_scalar(
                    'val/output_' + k, v, self.global_step)
        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.tensor([x['psnr'] for x in outputs]).mean()

        self.log('val/psnr', mean_psnr, prog_bar=True)