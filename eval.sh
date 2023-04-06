#!/bin/bash

EXP=$1
NAME=$2
DATA_DIR=/media/public_dataset/NeRF/nerf_synthetic/$1

DIR=/media/NFS/fong/refnerf-pytorch-pl
cd ${DIR}

RENDER_CHUNK_SIZE=4096

DISP_METRICS=True
NORMAL_METRICS=True

python3 eval.py \
  --gin_configs="${DIR}/exps/logs/${EXP}/${EXP}_${NAME}/config.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/exps'" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="Config.compute_disp_metrics = $DISP_METRICS" \
  --gin_bindings="Config.compute_normal_metrics = $NORMAL_METRICS" \