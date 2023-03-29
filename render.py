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

"""Render script."""

import concurrent.futures
import functools
import glob
import os
import time
import gc

from absl import app
import torch
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import utils

from matplotlib import cm
import mediapy as media
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from internal.nerf_system import RefNeRFSystem
from dataclasses import asdict

configs.define_common_flags()

def main(unused_argv):

    config = configs.load_config(save_config=False)
    hparams = asdict(config)

    summary_writer = SummaryWriter(
        os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name, 'test'))

    # Setup device.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create test dataset.
    dataset = datasets.load_dataset('test', config.data_dir, config)

    # Set random number generator seeds.
    torch.manual_seed(20221019)
    np.random.seed(20221019)


    # Load states from checkpoint.
    if utils.isdir(config.checkpoint_dir):    
        files = sorted([f for f in os.listdir(os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name))
                        if f.endswith('.ckpt')], key=lambda x: 1e10 if x.split('=')[-1][:-5]=='last'\
                            else int(x.split('=')[-1][:-5]))

        # if there are checkpoints in the dir, load the latest checkpoint
        if files:
            checkpoint_name = files[-1]
        else:
            raise ValueError("No checkpoints.")
    else:
        raise ValueError("Wrong checkpoint_dir.")
    
    # reload state
    system = RefNeRFSystem.load_from_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name, checkpoint_name), hparams=hparams, config=config, summary_writer=summary_writer)


    step = checkpoint_name.split('=')[-1][:-5]
    print(f'Rendering checkpoint at step {step}.')

    out_name = 'path_renders' if config.render_path else 'test_preds'
    out_name = f'{out_name}_step_{step}'
    base_dir = config.render_dir
    if base_dir is None:
        base_dir = os.path.join(config.checkpoint_dir, 'render')
    out_dir = os.path.join(base_dir, out_name)
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)

    system.render(dataset, base_dir, out_dir, out_name)

if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)
