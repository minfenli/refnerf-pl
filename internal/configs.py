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

"""Utility functions for handling configurations."""

import dataclasses
from typing import Any, Callable, Optional, Tuple

from absl import flags
import gin
from internal import utils
import numpy as np
import os

gin.add_config_file_search_path('experimental/users/barron/mipnerf360/')

@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  exp_name: str = 'exp'
  seed: int = 20230227
  num_workers: int = 4
  num_gpus: int = 1
  val_sample_num: int = 3
  sample_angle_range: float = 5 

  dataset_loader: str = 'llff'  # The type of dataset loader to use.
  dataset_debug_mode: bool = False  # If True, always loads specific batch
  batching: str = 'all_images'  # Batch composition, [single_image, all_images].
  batch_size: int = 16384  # The number of rays/pixels in each batch.
  patch_size: int = 1  # Resolution of patches sampled for training batches.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  load_alphabetical: bool = True  # Load images in COLMAP vs alphabetical
  # ordering (affects heldout test set).
  forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
  render_path: bool = False  # If True, render a path. Used only by LLFF.
  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  # If true, use all input images for training.
  llff_use_all_images_for_training: bool = False
  use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
  compute_eval_metrics: bool = False  # If True, compute SSIM and PSNR
  compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
  compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
  gc_every: int = 10000  # The number of steps between garbage collections.
  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 2.  # Near plane distance.
  far: float = 6.  # Far plane distance.
  checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
  render_dir: Optional[str] = None  # Output rendering directory.
  data_dir: Optional[str] = None  # Input data directory.
  vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.
  render_chunk_size: int = 16384  # Chunk size for whole-image renderings.
  num_showcase_images: int = 5  # The number of test-set images to showcase.
  deterministic_showcase: bool = True  # If True, showcase the same images.
  vis_num_rays: int = 16  # The number of rays to visualize.
  # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.
  vis_decimate: int = 0
  save_top_k: int = 5
  resume_path = None

  # Only used by train.py:
  max_steps: int = 250000  # The number of optimization steps.
  early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
  checkpoint_every: int = 25000  # The number of steps to save a checkpoint.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  train_render_every: int = 5000  # Steps between test set renders when training
  cast_rays_in_train_step: bool = False  # If True, compute rays in train step.
  data_loss_type: str = 'charb'  # What kind of loss to use ('mse' or 'charb').
  charb_padding: float = 0.001  # The padding used for Charbonnier loss.
  data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
  data_coarse_loss_mult: float = 0.  # Multiplier for the coarser data terms.
  interlevel_loss_mult: float = 1.0  # Mult. for the loss on the proposal MLP.
  orientation_loss_mult: float = 0.0  # Multiplier on the orientation loss.
  orientation_coarse_loss_mult: float = 0.0  # Coarser orientation loss weights.
  # What that loss is imposed on, options are 'normals' or 'normals_pred'.
  orientation_loss_target: str = 'normals_pred'
  predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
  # Mult. on the coarser predicted normal loss.
  predicted_normal_coarse_loss_mult: float = 0.0

  sample_noise_size: int = 128  # The number of rays/pixels for noise sampling in each batch.
  consistency_normal_loss_mult: float = 0.0
  consistency_normal_coarse_loss_mult: float = 0.0
  consistency_diffuse_loss_mult: float = 0.0
  consistency_diffuse_coarse_loss_mult: float = 0.0
  consistency_specular_loss_mult: float = 0.0
  consistency_specular_coarse_loss_mult: float = 0.0

  lr_init: float = 0.002  # The initial learning rate.
  lr_final: float = 0.00002  # The final learning rate.
  lr_delay_steps: int = 512  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.01  # How much severe the "warmup" should be.
  adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
  adam_beta2: float = 0.999  # Adam's beta2 hyperparameter.
  adam_eps: float = 1e-6  # Adam's epsilon hyperparameter.
  grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
  distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.

  # Only used by eval.py:
  eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
  eval_save_output: bool = True  # If True save predicted images to disk.
  eval_save_ray_data: bool = False  # If True save individual ray traces.
  eval_render_interval: int = 1  # The interval between images saved to disk.
  eval_dataset_limit: int = np.iinfo(np.int32).max  # Num test images to eval.
  eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
  eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).

  # Only used by render.py
  render_video_fps: int = 60  # Framerate in frames-per-second.
  render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
  render_path_frames: int = 120  # Number of frames in render path.
  z_variation: float = 0.  # How much height variation in render path.
  z_phase: float = 0.  # Phase offset for height variation in render path.
  render_dist_percentile: float = 0.5  # How much to trim from near/far planes.
  render_dist_curve_fn: Callable[..., Any] = np.log  # How depth is curved.
  render_path_file: Optional[str] = None  # Numpy render pose file to load.
  render_job_id: int = 0  # Render job id.
  render_num_jobs: int = 1  # Total number of render jobs.
  render_resolution: Optional[Tuple[int, int]] = None  # Render resolution, as
  # (width, height).
  render_focal: Optional[float] = None  # Render focal length.
  render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
  render_spherical: bool = False  # Render spherical 360 panoramas.
  render_save_async: bool = True  # Save to CNS using a separate thread.

  render_spline_keyframes: Optional[str] = None  # Text file containing names of
  # images to be used as spline
  # keyframes, OR directory
  # containing those images.
  render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
  render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
  render_spline_smoothness: float = .03  # B-spline smoothing factor, 0 for

def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_string('mode', None, 'Required by GINXM, not used.')
  flags.DEFINE_string('base_folder', None, 'Required by GINXM, not used.')
  flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
  flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def load_config(save_config=True):
  """Load the config, and optionally checkpoint it."""
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True)
  config = Config()
  if save_config:
    dir = os.path.join(config.checkpoint_dir, 'logs', config.exp_name)
    utils.makedirs(dir)
    with utils.open_file(dir + '/config.gin', 'w') as f:
      f.write(gin.config_str())
  return config
