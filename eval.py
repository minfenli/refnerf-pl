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

"""Evaluation script."""

import functools
import os
from os import path
import sys
import time
from absl import app, flags
import torch
from torch.utils.tensorboard import SummaryWriter
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import ref_utils
from internal import train_utils
from internal import utils
from internal import vis
import numpy as np


configs.define_common_flags()
FLAGS = flags.FLAGS


def main(unused_argv):
    config = configs.load_config(save_config=False)

    # setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    dataset = datasets.load_dataset('test', config.data_dir, config)
    setup = train_utils.setup_model(config, dataset=dataset)
    model, _, _, render_eval_fn, _ = setup
    model.eval()
    state = dict(step=0, model=model.state_dict())

    cc_fun = image.color_correct

    metric_harness = image.MetricHarness()

    last_step = 0
    out_dir = os.path.join(config.checkpoint_dir,
                           'path_renders' if config.render_path else 'test_preds')

    def path_fn(x): return os.path.join(out_dir, x)

    if not config.eval_only_once:
        summary_writer = SummaryWriter(
            os.path.join(config.checkpoint_dir, 'eval'))

    while True:

        # load checkpoint from file if it exists
        files = sorted([f for f in os.listdir(config.checkpoint_dir)
                        if f.startswith('checkpoint')], key=lambda x: int(x.split('_')[-1]))
        # if there are checkpoints in the dir, load the latest checkpoint
        if not files:
            print(f'No checkpoints yet. Sleeping.')
            time.sleep(10)
            continue

        # reload state
        checkpoint_name = files[-1]
        state = torch.load(os.path.join(
            config.checkpoint_dir, checkpoint_name))
        model.load_state_dict(state['model'])

        step = int(state['step'])

        if step <= last_step:
            print(
                f'Checkpoint step {step} <= last step {last_step}, sleeping.')
            time.sleep(10)
            continue
        print(f'Evaluating checkpoint at step {step}.')
        if config.eval_save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)

        num_eval = min(dataset.size, config.eval_dataset_limit)
        perm = torch.randperm(num_eval)
        showcase_indices = torch.sort(perm[:config.num_showcase_images])

        metrics = []
        metrics_cc = []
        showcases = []
        render_times = []

        # render and evaluate all test images
        for idx in range(dataset.size):
            eval_start_time = time.time()
            batch = next(dataset)
            if idx >= num_eval:
                print(f'Skipping image {idx+1}/{dataset.size}')
                continue
            print(f'Evaluating image {idx+1}/{dataset.size}')
            rays = batch.rays
            train_frac = state['step'] / config.max_steps
            with torch.no_grad():
                rendering = models.render_image(
                    functools.partial(render_eval_fn, train_frac), rays, config)

            render_times.append((time.time() - eval_start_time))
            print(f'Rendered in {render_times[-1]:0.3f}s')

            # Cast to 64-bit to ensure high precision for color correction function.
            gt_rgb = torch.tensor(
                batch.rgb, dtype=torch.float64, device=torch.device('cpu'))

            # move renderings to cpu to allow for metrics calculations
            rendering = {k: v.cpu().double() for k, v in rendering.items() if not k.startswith('ray_')}

            cc_start_time = time.time()
            rendering['rgb_cc'] = cc_fun(rendering['rgb'], gt_rgb)
            print(f'Color corrected in {(time.time() - cc_start_time):0.3f}s')

            if not config.eval_only_once and idx in showcase_indices:
                showcase_idx = idx if config.deterministic_showcase else len(
                    showcases)
                showcases.append((showcase_idx, rendering, batch))
            if not config.render_path:
                rgb = rendering['rgb']
                rgb_cc = rendering['rgb_cc']
                rgb_gt = gt_rgb

                if config.eval_quantize_metrics:
                    # Ensures that the images written to disk reproduce the metrics.
                    rgb = np.round(rgb * 255) / 255
                    rgb_cc = np.round(rgb_cc * 255) / 255

                if config.eval_crop_borders > 0:
                    def crop_fn(
                        x, c=config.eval_crop_borders): return x[c:-c, c:-c]
                    rgb = crop_fn(rgb)
                    rgb_cc = crop_fn(rgb_cc)
                    rgb_gt = crop_fn(rgb_gt)

                # calculate PSNR and SSIM metrics between rendering and gt
                metric = metric_harness(rgb, rgb_gt)
                metric_cc = metric_harness(rgb_cc, rgb_gt)

                if config.compute_disp_metrics:
                    for tag in ['mean', 'median']:
                        key = f'distance_{tag}'
                        if key in rendering:
                            disparity = 1 / (1 + rendering[key])
                            metric[f'disparity_{tag}_mse'] = float(
                                ((disparity - batch.disps)**2).mean())

                if config.compute_normal_metrics:
                    weights = rendering['acc'] * batch.alphas
                    normalized_normals_gt = ref_utils.l2_normalize(
                        batch.normals)
                    for key, val in rendering.items():
                        if key.startswith('normals') and val is not None:
                            normalized_normals = ref_utils.l2_normalize(val)
                            metric[key + '_mae'] = ref_utils.compute_weighted_mae(
                                weights, normalized_normals, normalized_normals_gt)

                for m, v in metric.items():
                    print(f'{m:30s} = {v:.4f}')

                metrics.append(metric)
                metrics_cc.append(metric_cc)

            if config.eval_save_output and (config.eval_render_interval > 0):
                if (idx % config.eval_render_interval) == 0:
                    utils.save_img_u8(rendering['rgb'],
                                      path_fn(f'color_{idx:03d}.png'))
                    utils.save_img_u8(rendering['rgb_cc'],
                                      path_fn(f'color_cc_{idx:03d}.png'))

                    for key in ['distance_mean', 'distance_median']:
                        if key in rendering:
                            utils.save_img_f32(rendering[key],
                                               path_fn(f'{key}_{idx:03d}.tiff'))

                    for key in ['normals']:
                        if key in rendering:
                            utils.save_img_u8(rendering[key] / 2. + 0.5,
                                              path_fn(f'{key}_{idx:03d}.png'))

                    utils.save_img_f32(
                        rendering['acc'], path_fn(f'acc_{idx:03d}.tiff'))

        if not config.eval_only_once:
            summary_writer.add_scalar(
                'eval_median_render_time', np.median(render_times), step)
            for name in metrics[0]:
                scores = [m[name] for m in metrics]
                summary_writer.add_scalar(
                    'eval_metrics/' + name, np.mean(scores), step)
                summary_writer.add_histogram(
                    'eval_metrics/' + 'perimage_' + name, scores, step)
            for name in metrics_cc[0]:
                scores = [m[name] for m in metrics_cc]
                summary_writer.add_scalar(
                    'eval_metrics_cc/' + name, np.mean(scores), step)
                summary_writer.add_histogram(
                    'eval_metrics_cc/' + 'perimage_' + name, scores, step)

            for i, r, b in showcases:
                if config.vis_decimate > 1:
                    d = config.vis_decimate

                    def decimate_fn(x, d=d):
                        return None if x is None else x[::d, ::d]
                else:
                    def decimate_fn(x): return x
                r = decimate_fn(r)
                b = decimate_fn(b)
                visualizations = vis.visualize_suite(r, b.rays)
                for k, v in visualizations.items():
                    summary_writer.image(f'output_{k}_{i}', v, step)
                if not config.render_path:
                    target = b.rgb
                    summary_writer.image(f'true_color_{i}', target, step)
                    pred = visualizations['color']
                    residual = np.clip(pred - target + 0.5, 0, 1)
                    summary_writer.image(f'true_residual_{i}', residual, step)
                    if config.compute_normal_metrics:
                        summary_writer.image(f'true_normals_{i}', b.normals / 2. + 0.5,
                                             step)

        if (config.eval_save_output and not config.render_path):
            with utils.open_file(path_fn(f'render_times_{step}.txt'), 'w') as f:
                f.write(' '.join([str(r) for r in render_times]))
            for name in metrics[0]:
                with utils.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
                    f.write(' '.join([str(m[name]) for m in metrics]))
            for name in metrics_cc[0]:
                with utils.open_file(path_fn(f'metric_cc_{name}_{step}.txt'), 'w') as f:
                    f.write(' '.join([str(m[name]) for m in metrics_cc]))
            if config.eval_save_ray_data:
                for i, r, b in showcases:
                    rays = {k: v for k, v in r.items() if 'ray_' in k}
                    np.set_printoptions(threshold=sys.maxsize)
                    with utils.open_file(path_fn(f'ray_data_{step}_{i}.txt'), 'w') as f:
                        f.write(repr(rays))
            # import pdb; pdb.set_trace()
            with utils.open_file(path_fn(f'avg_metrics_{step}.txt'), 'w') as f:
                f.write(f'render_time: {np.mean(render_times)}\n')
                for name in metrics[0]:
                    f.write(f'{name}: {np.mean([m[name] for m in metrics])}\n')
                for name in metrics_cc[0]:
                    f.write(
                        f'cc_{name}: {np.mean([m[name] for m in metrics_cc])}\n')

        if config.eval_only_once:
            break
        if config.early_exit_steps is not None:
            num_steps = config.early_exit_steps
        else:
            num_steps = config.max_steps
        if int(step) >= num_steps:
            break
        last_step = step


if __name__ == '__main__':
    with gin.config_scope('eval'):
        app.run(main)
