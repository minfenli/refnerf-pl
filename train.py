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

"""Training script."""

import os
import sys
import numpy as np
import random
import torch
from absl import flags

from torch.utils.tensorboard import SummaryWriter
import gin.torch
from internal import configs
from internal.nerf_system import RefNeRFSystem

from dataclasses import asdict

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

configs.define_common_flags()
FLAGS = flags.FLAGS

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(unused_argv):
    # load config file and save params to checkpoint folder
    config = configs.load_config()
    hparams = asdict(config)

    # set random seeds for reproducibility
    setup_seed(config.seed)

    summary_writer = SummaryWriter(os.path.join(config.checkpoint_dir, 
                                                'logs', 
                                                config.exp_name.split('_')[0], 
                                                config.exp_name))

    system = RefNeRFSystem(hparams, config, summary_writer)

    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(config.checkpoint_dir, 
                                                   'ckpt', 
                                                   config.exp_name.split('_')[0],
                                                   config.exp_name),
                              save_last=True,
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=config.save_top_k,
                              )
                              
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]


    trainer = Trainer(
        max_steps=config.max_steps,
        max_epochs=-1,
        callbacks=callbacks,
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
    
    trainer.fit(system, ckpt_path=config.resume_path)



if __name__ == '__main__':
    with gin.config_scope('train'):
        FLAGS(sys.argv)
        main(sys.argv)
