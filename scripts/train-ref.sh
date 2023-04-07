#!/bin/bash

EXP=$1
NAME=$2
CONFIG_PATH=$3
DATA_DIR=/media/public_dataset/NeRF/ref/ref/$1

DIR=/media/NFS/fong/refnerf-pl
cd ${DIR}

MAX_STEPS=250000
VAL_EVERY=10000

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

python3 train.py \
  --gin_configs="${CONFIG_PATH}" \
  --gin_bindings="Config.exp_name = '${EXP}_${NAME}'" \
  --gin_bindings="Config.max_steps = ${MAX_STEPS}" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/exps'" \
  --gin_bindings="Config.checkpoint_every = ${VAL_EVERY}" \