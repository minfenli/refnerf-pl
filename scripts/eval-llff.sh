#!/bin/bash

EXP=$1
NAME=$2
DATA_DIR=/media/public_dataset/NeRF/nerf_llff_data/$1

DIR=/media/NFS/fong/refnerf-pytorch-pl
cd ${DIR}

RENDER_CHUNK_SIZE=4096

python3 eval.py \
  --gin_configs="${DIR}/exps/logs/${EXP}/${EXP}_${NAME}/config.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/exps'" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \