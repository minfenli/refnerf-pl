#!/bin/bash

EXP=$1
NAME=$2
CONFIG=$3
DATA_DIR=/media/public_dataset/NeRF/nerf_synthetic/$1

DIR=/media/NFS/fong/refnerf-pytorch-pl
cd ${DIR}

BATCH_SIZE=1024
RENDER_CHUNK_SIZE=4096
MAX_STEPS=300000
VAL_EVERY=10000

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

# If job gets evicted reload generated config file not original that might have been modified
if [ -f "${DIR}/exps/ckpt/${NAME}/config.gin"]; then
  CONFIG_PATH="${DIR}/exps/ckpt/${NAME}/config.gin"
else
  CONFIG_PATH="$CONFIG"
fi
python3 train.py \
  --gin_configs="${CONFIG_PATH}" \
  --gin_bindings="Config.exp_name = '${EXP}_${NAME}'" \
  --gin_bindings="Config.max_steps = ${MAX_STEPS}" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/exps'" \
  --gin_bindings="Config.batch_size = ${BATCH_SIZE}" \
  --gin_bindings="Config.render_chunk_size = ${RENDER_CHUNK_SIZE}" \
  --gin_bindings="Config.checkpoint_every = ${VAL_EVERY}" \