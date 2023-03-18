# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple
import torch

from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import stepfun
from internal import utils


def compute_data_loss(batch, renderings, rays, config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = rays.lossmult
    lossmult = torch.broadcast_to(lossmult, batch.rgb[..., :3].shape)
    if config.disable_multiscale_loss:
        lossmult = torch.ones_like(lossmult)
    for rendering in renderings:
        gt_rgb = torch.tensor(batch.rgb[..., :3], device=rendering['rgb'].device)
        if config.supervised_by_linear_rgb:
            gt_rgb = image.srgb_to_linear(gt_rgb)
        resid_sq = (rendering['rgb'] - gt_rgb)**2
        denom = lossmult.sum()
        stats['mses'].append((lossmult * resid_sq).sum() / denom)

        if config.data_loss_type == 'mse':
            # Mean-squared error (L2) loss.
            data_loss = resid_sq
        elif config.data_loss_type == 'charb':
            # Charbonnier loss.
            data_loss = torch.sqrt(resid_sq + config.charb_padding**2)
        else:
            assert False
        data_losses.append((lossmult * data_loss).sum() / denom)

        if config.compute_disp_metrics:
            # Using mean to compute disparity, but other distance statistics can
            # be used instead.
            disp = 1 / (1 + rendering['distance_mean'])
            stats['disparity_mses'].append(((disp - batch.disps)**2).mean())

        if config.compute_normal_metrics:
            if 'normals' in rendering:
                weights = rendering['acc'] * batch.alphas
                normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
                normalized_normals = ref_utils.l2_normalize(
                    rendering['normals'])
                normal_mae = ref_utils.compute_weighted_mae(weights, normalized_normals,
                                                            normalized_normals_gt)
            else:
                # If normals are not computed, set MAE to NaN.
                normal_mae = torch.nan

            stats['normal_maes'].append(normal_mae)

    data_losses = torch.stack(data_losses)
    loss = \
        config.data_coarse_loss_mult * torch.sum(data_losses[:-1]) + \
        config.data_loss_mult * data_losses[-1]
    stats = {k: torch.tensor(stats[k], device=loss.device) for k in stats}
    return loss, stats  

def compute_depth_smoothness_loss(renderings, config, geometry_warmup_ratio):
    """Computes smoothness loss terms for depth outputs relating RGB edges."""
    smoothness_losses = []

    loss = lambda x: torch.mean(torch.abs(x))
    bilateral_filter = lambda x: torch.exp(-torch.abs(x).mean(-1, keepdim=True))


    for rendering in renderings:
        depths = rendering['distance']

        with torch.no_grad():
            acc00 = rendering['acc'][...,:-1,:-1,None]
            weights = rendering['rgb']

        v00 = depths[...,:-1,:-1,:]
        v01 = depths[...,:-1,1:,:]
        v10 = depths[...,1:,:-1,:]

        w01 = bilateral_filter(weights[...,:-1,:-1,:] - weights[...,:-1,1:,:])
        w10 = bilateral_filter(weights[...,:-1,:-1,:] - weights[...,1:,:-1,:])
        L1 = loss(acc00 * w01 * (v00 - v01)**2)
        L2 = loss(acc00 * w10 * (v00 - v10)**2)
        smoothness_losses.append((L1 + L2) / 2)

    smoothness_losses = torch.stack(smoothness_losses)

    loss = geometry_warmup_ratio * \
        (config.depth_smoothness_coarse_loss_mult * torch.sum(smoothness_losses[:-1]) + \
        config.depth_smoothness_loss_mult * smoothness_losses[-1])
    return loss
    
# def compute_depth_smoothness_loss(renderings, config, geometry_warmup_ratio):
#     """Computes smoothness loss terms for depth outputs relating RGB edges."""
#     smoothness_losses = []

#     loss = lambda x: torch.mean(torch.abs(x))
#     bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1, keepdim=True) * config.depth_smoothness_gamma)


#     for rendering in renderings:
#         inputs = rendering['distance']
#         weights = rendering['rgb']

#         w1 = bilateral_filter(weights[...,:,:-1,:] - weights[...,:,1:,:])
#         w2 = bilateral_filter(weights[...,:-1,:,:] - weights[...,1:,:,:])
#         w3 = bilateral_filter(weights[...,:-1,:-1,:] - weights[...,1:,1:,:])
#         w4 = bilateral_filter(weights[...,1:,:-1,:] - weights[...,:-1,1:,:])
        
#         L1 = loss(w1 * (inputs[...,:,:-1,:] - inputs[...,:,1:,:]))
#         L2 = loss(w2 * (inputs[...,:-1,:,:] - inputs[...,1:,:,:]))
#         L3 = loss(w3 * (inputs[...,:-1,:-1,:] - inputs[...,:,1:,1:,:]))
#         L4 = loss(w4 * (inputs[...,1:,:-1,:] - inputs[...,:-1,1:,:]))
#         smoothness_losses.append((L1 + L2 + L3 + L4) / 4)

#     smoothness_losses = torch.stack(smoothness_losses)

#     loss = geometry_warmup_ratio * \
#         (config.depth_smoothness_coarse_loss_mult * torch.sum(smoothness_losses[:-1]) + \
#         config.depth_smoothness_loss_mult * smoothness_losses[-1])
#     return loss

def interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    loss_interlevel = 0.
    for ray_results in ray_history[:-1]:
        cp = ray_results['sdist']
        wp = ray_results['weights']
        loss_interlevel += torch.mean(stepfun.lossfun_outer(c, w, cp, wp))
    return config.interlevel_loss_mult * loss_interlevel


def orientation_loss(rays, model, ray_history, config, geometry_warmup_ratio):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    total_loss = 0.
    zero = torch.tensor(0.0, dtype=torch.float32, device=rays.viewdirs.device)
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results[config.orientation_loss_target]
        if n is None:
            raise ValueError(
                'Normals cannot be None if orientation loss is on.')
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -rays.viewdirs
        n_dot_v = (n * v[..., None, :]).sum(dim=-1)
        loss = torch.mean((w * torch.minimum(zero, n_dot_v)**2).sum(dim=-1))
        if i < model.num_levels - 1:
            total_loss += geometry_warmup_ratio * config.orientation_coarse_loss_mult * loss
        else:
            total_loss += geometry_warmup_ratio * config.orientation_loss_mult * loss
    return total_loss


def predicted_normal_loss(model, ray_history, config, geometry_warmup_ratio):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        # with torch.no_grad():
        n = ray_results['normals']
        n_pred = ray_results['normals_pred']
        if n is None or n_pred is None:
            raise ValueError(
                'Predicted normals and gradient normals cannot be None if '
                'predicted normal loss is on.')
        loss = torch.mean(
            (w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
        if i < model.num_levels - 1:
            total_loss += geometry_warmup_ratio * config.predicted_normal_coarse_loss_mult * loss
        else:
            total_loss += geometry_warmup_ratio * config.predicted_normal_loss_mult * loss
    return total_loss


def noisy_consistency_loss(model, renderings, renderings_noise, config, warmup_ratio=1.):
    """Computes the consistency loss."""
    total_diffuse_loss = 0.
    total_specular_loss = 0.
    total_normal_loss = 0.
    n_samples = config.sample_noise_size // config.patch_size**2
    n_angles = config.sample_noise_angles

    for i, (rendering, rendering_noise) in enumerate(zip(renderings, renderings_noise)):
        # (n_samples, n_angles, ...)
        noise_diffuse_rgb = rendering_noise['diffuse'].reshape(n_samples, n_angles, *rendering_noise['diffuse'].shape[1:])
        noise_specular_rgb = rendering_noise['specular'].reshape(n_samples, n_angles, *rendering_noise['specular'].shape[1:])
        
        if config.consistency_diffuse_loss_type == 'mse':
            # (n_samples, n_angles, ...)
            diffuse_mse = (rendering['diffuse'][:n_samples, None] - noise_diffuse_rgb)**2
            diffuse_loss = diffuse_mse.sum(axis=-1).mean()
        elif config.consistency_diffuse_loss_type == 'avg_mse':
            # (n_samples, n_angles, ...)
            diffuse_mse = (rendering['diffuse'][:n_samples, None] - noise_diffuse_rgb.mean(axis=1, keepdim=True))**2
            diffuse_loss = diffuse_mse.sum(axis=-1).mean()
        elif config.consistency_specular_loss_type == 'var':
            diffuse_rays = torch.cat([rendering['diffuse'][:n_samples, None], noise_diffuse_rgb], axis=1)            
            diffuse_var = diffuse_rays.var(axis=1, keepdim=True).mean(axis=-1, keepdim=True)
            diffuse_loss = diffuse_var.sum(axis=-1).mean()

        if config.consistency_specular_loss_type == 'mse':
            specular_mse = (rendering['specular'][:n_samples, None] - noise_specular_rgb)**2
            specular_loss = -specular_mse.sum(axis=-1).mean()
        elif config.consistency_specular_loss_type == 'avg_mse':
            specular_mse = (rendering['specular'][:n_samples, None] - noise_specular_rgb.mean(axis=1, keepdim=True))**2
            specular_loss = -specular_mse.sum(axis=-1).mean()
        elif config.consistency_specular_loss_type == 'var':
            specular_rays = torch.cat([rendering['specular'][:n_samples, None], noise_specular_rgb], axis=1)            
            specular_var = specular_rays.var(axis=1, keepdim=True).mean(axis=-1, keepdim=True)
            specular_loss = -specular_var.sum(axis=-1).mean()

        n = rendering['normals'][:n_samples, None]
        n_pred = rendering['normals_pred'][:n_samples, None]
        n_noise = rendering_noise['normals'].reshape(n_samples, n_angles, *rendering_noise['normals'].shape[1:])
        n_pred_noise = rendering_noise['normals_pred'].reshape(n_samples, n_angles, *rendering_noise['normals_pred'].shape[1:])

        if n is None or n_pred is None:
            raise ValueError(
                'Predicted normals and gradient normals cannot be None if '
                'consistency loss is on.')

        if config.consistency_normal_loss_target == 'normals':                
            normal_loss = torch.mean((1.0 - torch.sum(n * n_noise, dim=-1))) + \
                torch.mean((1.0 - torch.sum(n * n_noise, dim=-1)))
        elif config.consistency_normal_loss_target == 'normals_pred': 
            normal_loss = torch.mean((1.0 - torch.sum(n * n_noise, dim=-1))) + \
                torch.mean((1.0 - torch.sum(n_pred * n_pred_noise, dim=-1)))   
        else:
            raise ValueError(
                'Given an unknown type of consistency_normal_loss_target.')

        if i < model.num_levels - 1:
            total_diffuse_loss += warmup_ratio * config.consistency_diffuse_coarse_loss_mult * diffuse_loss
            total_specular_loss += warmup_ratio * config.consistency_specular_coarse_loss_mult * specular_loss
            total_normal_loss += warmup_ratio * config.consistency_normal_coarse_loss_mult * normal_loss
        else:
            total_diffuse_loss += warmup_ratio * config.consistency_diffuse_loss_mult * diffuse_loss
            total_specular_loss += warmup_ratio * config.consistency_specular_loss_mult * specular_loss
            total_normal_loss += warmup_ratio * config.consistency_normal_loss_mult * normal_loss
    return total_diffuse_loss, total_specular_loss, total_normal_loss


def accumulated_weights_loss(renderings, config):
    """Computes accumulated_weights_loss to intrigue model output higher accs."""
    return config.accumulated_weights_loss_mult * \
        ((1-renderings[-1]['acc'])**2).mean()


def create_train_step(model: models.Model,
                      config: configs.Config,
                      dataset: Optional[datasets.NeRFDataset] = None):
    """Creates the pmap'ed Nerf training function.

    Args:
      model: The linen model.
      config: The configuration.
      dataset: Training dataset.

    Returns:
      training function.
    """
    if dataset is None:
        camtype = camera_utils.ProjectionType.PERSPECTIVE
    else:
        camtype = dataset.camtype

    def train_step(
        model,
        optimizer,
        lr_scheduler,
        batch,
        cameras,
        train_frac,
    ):
        """One optimization step.

        Args:
          state: TrainState, state of the model/optimizer.
          batch: dict, a mini-batch of data for training.
          cameras: module containing camera poses.
          train_frac: float, the fraction of training that is complete.

        Returns:
          A tuple (new_state, stats) with
            new_state: TrainState, new training state.
            stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        """
        rays = batch.rays
        if config.cast_rays_in_train_step:
            rays = camera_utils.cast_ray_batch(
                cameras, rays, camtype, xnp=torch).to(model.device)
        else:
            rays.to(model.device)

        # clear gradients
        optimizer.zero_grad()

        renderings, ray_history = model(
            rays,
            train_frac=train_frac,
            compute_extras=\
                config.compute_disp_metrics or config.compute_normal_metrics)

        losses = {}

        # calculate photometric error
        data_loss, stats = compute_data_loss(batch, renderings, rays, config)
        losses['data'] = data_loss

        # calculate interlevel loss
        if config.interlevel_loss_mult > 0:
            losses['interlevel'] = interlevel_loss(ray_history, config)

        # calculate normals orientation loss
        if (config.orientation_coarse_loss_mult > 0 or
                config.orientation_loss_mult > 0):
            losses['orientation'] = orientation_loss(
                rays, model, ray_history, config)

        # calculate predicted normal loss
        if (config.predicted_normal_coarse_loss_mult > 0 or
                config.predicted_normal_loss_mult > 0):
            losses['predicted_normals'] = predicted_normal_loss(
                model, ray_history, config)

        params = dict(model.named_parameters())
        stats['weights_l2s'] = {k.replace('.', '/') : params[k].detach().norm() ** 2 for k in params}

        # calculate total loss
        loss = torch.sum(torch.stack(list(losses.values())))
        stats['loss'] = loss.detach().cpu()
        stats['losses'] = {key: losses[key].detach().cpu() for key in losses}

        # backprop
        loss.backward()

        # calculate average grad and stats
        stats['grad_norms'] = {k.replace('.', '/') : params[k].grad.detach().cpu().norm() for k in params}
        stats['grad_maxes'] = {k.replace('.', '/') : params[k].grad.detach().cpu().abs().max() for k in params}

        # Clip gradients
        if config.grad_max_val > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.grad_max_val)
        if config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_max_norm)

        #TODO: set nan grads to 0

        # update the model weights
        optimizer.step()

        # update learning rate
        lr_scheduler.step()

        #TODO: difference between previous and current state - Redundant?
        # stats['opt_update_norms'] = summarize_tree(opt_delta, tree_norm)
        # stats['opt_update_maxes'] = summarize_tree(opt_delta, tree_abs_max)

        # Calculate PSNR metric
        stats['psnrs'] = image.mse_to_psnr(stats['mses'])
        stats['psnr'] = stats['psnrs'][-1]

        # return new state and statistics
        return stats

    return train_step


def create_optimizer(
        config: configs.Config,
        params: Dict) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Creates optimizer for model training."""
    adam_kwargs = {
        'lr': config.lr_init,
        'betas': (config.adam_beta1, config.adam_beta2),
        'eps': config.adam_eps,
    }
    lr_kwargs = {
        'lr_init': config.lr_init,
        'lr_final': config.lr_final,
        'max_steps': config.max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
    }
    optimizer = torch.optim.Adam(params=params, **adam_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(math.learning_rate_decay, **lr_kwargs))
    return optimizer, lr_scheduler


def create_render_fn(model: models.Model):
    """Creates a function for full image rendering."""
    def render_eval_fn(train_frac, rays):
        return model(
            rays,
            train_frac=train_frac,
            compute_extras=True)
    return render_eval_fn


def setup_model(
        config: configs.Config,
        dataset: Optional[datasets.NeRFDataset] = None,
    ):
    """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

    dummy_rays = utils.dummy_rays()
    model = models.construct_model(dummy_rays, config)

    optimizer, lr_scheduler = create_optimizer(config, model.parameters())
    render_eval_fn = create_render_fn(model)
    train_step = create_train_step(model, config, dataset=dataset)

    return model, optimizer, lr_scheduler, render_eval_fn, train_step
