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

from internal.nerf_system import RefNeRFSystem
from dataclasses import asdict

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

configs.define_common_flags()
FLAGS = flags.FLAGS


def main(unused_argv):
    config = configs.load_config(save_config=False)
    hparams = asdict(config)

    summary_writer = SummaryWriter(
        os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name, 'test'))

    files = sorted([f for f in os.listdir(os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name))
                    if f.endswith('.ckpt')], key=lambda x: 1e10 if x.split('=')[-1][:-5]=='last'\
                        else int(x.split('=')[-1][:-5]))
    # if there are checkpoints in the dir, load the latest checkpoint
    if not files:
        print(f'No checkpoints yet.')
        return

    # reload state
    checkpoint_name = files[-1]
    system = RefNeRFSystem.load_from_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt', config.exp_name, checkpoint_name), hparams=hparams, config=config, summary_writer=summary_writer)

    trainer = Trainer(
        max_steps=config.max_steps,
        max_epochs=-1,
        val_check_interval=config.checkpoint_every,
        logger=None,
        enable_model_summary=False,
        accelerator='auto',
        devices=config.num_gpus,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if config.num_gpus == 1 else None,
        strategy=DDPPlugin(find_unused_parameters=False) if config.num_gpus > 1 else None,
        limit_val_batches=config.val_sample_num
    )

    # test (pass in the model)
    trainer.test(system)

if __name__ == '__main__':
    with gin.config_scope('eval'):
        app.run(main)
