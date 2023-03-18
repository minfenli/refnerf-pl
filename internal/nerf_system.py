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
from internal import ref_utils

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

import os
import sys

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
    
    def test_dataloader(self):
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
            consistency_warmup_ratio = min(1., self.global_step/(self.config.consistency_warmup_steps*self.config.max_steps))
        else:
            consistency_warmup_ratio = 1.
        
        if self.config.geometry_warmup_steps > 0.:
            geometry_warmup_ratio = min(1., self.global_step/(self.config.geometry_warmup_steps*self.config.max_steps))
        else:
            geometry_warmup_ratio = 1.

        if self.config.sample_noise_size > 0 and (
           self.config.consistency_diffuse_coarse_loss_mult > 0 or
           self.config.consistency_specular_coarse_loss_mult > 0 or 
           self.config.consistency_normal_coarse_loss_mult > 0 or
           self.config.consistency_diffuse_loss_mult > 0 or
           self.config.consistency_specular_loss_mult > 0 or 
           self.config.consistency_normal_loss_mult > 0):
            if self.config.patch_size**2 > self.config.sample_noise_size:
                raise ValueError(f'Patch size {self.config.patch_size}^2 too large for ' +
                                f'sampling noise view points {self.config.sample_noise_size}')
            sample_noise_size = self.config.sample_noise_size // self.config.patch_size**2
            noisy_rays = sample_utils.sample_noisy_rays(
                rays, renderings[-1], self.config.sample_angle_range, 
                sample_noise_size, self.config.sample_noise_angles, consistency_warmup_ratio)
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
                rays, self.model, ray_history, self.config, geometry_warmup_ratio)

        # calculate predicted normal loss
        if (self.config.predicted_normal_coarse_loss_mult > 0 or
                self.config.predicted_normal_loss_mult > 0):
            losses['predicted_normals'] = train_utils.predicted_normal_loss(
                self.model, ray_history, self.config, geometry_warmup_ratio)
        
        # calculate predicted normal loss
        if (self.config.patch_size > 1 and (
            self.config.depth_smoothness_coarse_loss_mult > 0 or
            self.config.depth_smoothness_loss_mult > 0)):
            losses['smoothness'] = train_utils.compute_depth_smoothness_loss(
                    renderings, self.config, geometry_warmup_ratio)

        # calculate predicted consistency loss
        if self.config.sample_noise_size > 0 and (
           self.config.consistency_diffuse_coarse_loss_mult > 0 or
           self.config.consistency_specular_coarse_loss_mult > 0 or 
           self.config.consistency_normal_coarse_loss_mult > 0 or
           self.config.consistency_diffuse_loss_mult > 0 or
           self.config.consistency_specular_loss_mult > 0 or 
           self.config.consistency_normal_loss_mult > 0):
            consistency_losses = train_utils.noisy_consistency_loss(
                self.model, renderings, renderings_noise, self.config, consistency_warmup_ratio)
            (losses['diffuse_consistency'], 
             losses['specular_consistency'], 
             losses['normals_consistency']) = consistency_losses

        # calculate accumulated weights loss
        if self.config.accumulated_weights_loss_mult > 0:
            losses['acc'] = train_utils.accumulated_weights_loss(renderings, self.config)

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
        pass

    def on_test_start(self):
        num_eval = min(self.val_dataset.size, self.config.eval_dataset_limit)
        perm = torch.randperm(num_eval)
        self.showcase_indices = torch.sort(perm[:self.config.num_showcase_images])

        self.metrics = []
        self.metrics_cc = []
        self.showcases = []
        self.render_times = []

        out_dir = os.path.join(self.config.checkpoint_dir, 'ckpt', self.config.exp_name,
                           'path_renders' if self.config.render_path else 'test_preds')

        def path_fn(x): return os.path.join(out_dir, x)

        self.path_fn = path_fn

    def on_test_end(self):
        if not self.config.eval_only_once:
            self.summary_writer.add_scalar(
                'eval_median_render_time', np.median(self.render_times), self.global_step)
            for name in self.metrics[0]:
                scores = [m[name] for m in self.metrics]
                self.summary_writer.add_scalar(
                    'eval_metrics/' + name, np.mean(scores), self.global_step)
                self.summary_writer.add_histogram(
                    'eval_metrics/' + 'perimage_' + name, scores, self.global_step)
            for name in self.metrics_cc[0]:
                scores = [m[name] for m in self.metrics_cc]
                self.summary_writer.add_scalar(
                    'eval_metrics_cc/' + name, np.mean(scores), self.global_step)
                self.summary_writer.add_histogram(
                    'eval_metrics_cc/' + 'perimage_' + name, scores, self.global_step)

            for i, r, b in self.showcases:
                if self.config.vis_decimate > 1:
                    d = self.config.vis_decimate

                    def decimate_fn(x, d=d):
                        return None if x is None else x[::d, ::d]
                else:
                    def decimate_fn(x): return x
                r = decimate_fn(r)
                b = decimate_fn(b)
                visualizations = vis.visualize_suite(r, b.rays)
                for k, v in visualizations.items():
                    self.summary_writer.image(f'output_{k}_{i}', v, self.global_step)
                if not self.config.render_path:
                    target = b.rgb
                    self.summary_writer.image(f'true_color_{i}', target, self.global_step)
                    pred = visualizations['color']
                    residual = np.clip(pred - target + 0.5, 0, 1)
                    self.summary_writer.image(f'true_residual_{i}', residual, self.global_step)
                    if self.config.compute_normal_metrics:
                        self.summary_writer.image(f'true_normals_{i}', b.normals / 2. + 0.5,
                                                self.global_step)

        if (self.config.eval_save_output and not self.config.render_path):
            with utils.open_file(self.path_fn(f'render_times_{self.global_step}.txt'), 'w') as f:
                f.write(' '.join([str(r) for r in self.render_times]))
            for name in self.metrics[0]:
                with utils.open_file(self.path_fn(f'metric_{name}_{self.global_step}.txt'), 'w') as f:
                    f.write(' '.join([str(m[name]) for m in self.metrics]))
            for name in self.metrics_cc[0]:
                with utils.open_file(self.path_fn(f'metric_cc_{name}_{self.global_step}.txt'), 'w') as f:
                    f.write(' '.join([str(m[name]) for m in self.metrics_cc]))
            if self.config.eval_save_ray_data:
                for i, r, b in self.showcases:
                    rays = {k: v for k, v in r.items() if 'ray_' in k}
                    np.set_printoptions(threshold=sys.maxsize)
                    with utils.open_file(self.path_fn(f'ray_data_{self.global_step}_{i}.txt'), 'w') as f:
                        f.write(repr(rays))
            # import pdb; pdb.set_trace()
            with utils.open_file(self.path_fn(f'avg_metrics_{self.global_step}.txt'), 'w') as f:
                f.write(f'render_time: {np.mean(self.render_times)}\n')
                for name in self.metrics[0]:
                    f.write(f'{name}: {np.mean([m[name] for m in self.metrics])}\n')
                for name in self.metrics_cc[0]:
                    f.write(
                        f'cc_{name}: {np.mean([m[name] for m in self.metrics_cc])}\n')

    
    def test_step(self, batch, batch_idx):
        cc_fun = image.color_correct
        with torch.no_grad():
            eval_start_time = time.time()
            (batch) = batch
            
            rays = batch.rays
            train_frac = self.global_step / self.config.max_steps
            
            rendering = models.render_image(
                functools.partial(self.render_eval_fn, train_frac), rays, self.config)

            self.render_times.append((time.time() - eval_start_time))

            # Cast to 64-bit to ensure high precision for color correction function.
            gt_rgb = torch.tensor(
                batch.rgb, dtype=torch.float64, device=torch.device('cpu'))

            # move renderings to cpu to allow for metrics calculations
            rendering = {k: v.cpu().double() for k, v in rendering.items() if not k.startswith('ray_')}

            cc_start_time = time.time()
            rendering['rgb_cc'] = cc_fun(rendering['rgb'], gt_rgb)
            if not self.config.eval_only_once and batch_idx in self.showcase_indices:
                showcase_idx = batch_idx if self.config.deterministic_showcase else len(
                    self.showcases)
                self.showcases.append((showcase_idx, rendering, batch))
            if not self.config.render_path:
                rgb = rendering['rgb']
                rgb_cc = rendering['rgb_cc']
                rgb_gt = gt_rgb

                if self.config.eval_quantize_metrics:
                    # Ensures that the images written to disk reproduce the metrics.
                    rgb = np.round(rgb * 255) / 255
                    rgb_cc = np.round(rgb_cc * 255) / 255

                if self.config.eval_crop_borders > 0:
                    def crop_fn(
                        x, c=self.config.eval_crop_borders): return x[c:-c, c:-c]
                    rgb = crop_fn(rgb)
                    rgb_cc = crop_fn(rgb_cc)
                    rgb_gt = crop_fn(rgb_gt)

                # calculate PSNR and SSIM metrics between rendering and gt
                metric = self.metric_harness(rgb, rgb_gt)
                metric_cc = self.metric_harness(rgb_cc, rgb_gt)

                if self.config.compute_disp_metrics:
                    for tag in ['mean', 'median']:
                        key = f'distance_{tag}'
                        if key in rendering:
                            disparity = 1 / (1 + rendering[key])
                            metric[f'disparity_{tag}_mse'] = float(
                                ((disparity - batch.disps)**2).mean())

                if self.config.compute_normal_metrics:
                    weights = rendering['acc'] * batch.alphas
                    normalized_normals_gt = ref_utils.l2_normalize(
                        batch.normals)
                    for key, val in rendering.items():
                        if key.startswith('normals') and val is not None:
                            normalized_normals = ref_utils.l2_normalize(val)
                            metric[key + '_mae'] = ref_utils.compute_weighted_mae(
                                weights, normalized_normals, normalized_normals_gt)

                self.metrics.append(metric)
                self.metrics_cc.append(metric_cc)

            if self.config.eval_save_output and (self.config.eval_render_interval > 0):
                if (batch_idx % self.config.eval_render_interval) == 0:
                    utils.save_img_u8(rendering['rgb'],
                                        self.path_fn(f'color_{batch_idx:03d}.png'))
                    utils.save_img_u8(rendering['rgb_cc'],
                                        self.path_fn(f'color_cc_{batch_idx:03d}.png'))

                    for key in ['distance_mean', 'distance_median']:
                        if key in rendering:
                            utils.save_img_f32(rendering[key],
                                                self.path_fn(f'{key}_{batch_idx:03d}.tiff'))

                    for key in ['normals']:
                        if key in rendering:
                            utils.save_img_u8(rendering[key] / 2. + 0.5,
                                                self.path_fn(f'{key}_{batch_idx:03d}.png'))

                    utils.save_img_f32(
                        rendering['acc'], self.path_fn(f'acc_{batch_idx:03d}.tiff'))
        return {}