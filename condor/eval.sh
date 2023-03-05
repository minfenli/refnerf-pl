#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refnerf

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"


DIR=/users/visics/gkouros/projects/nerf-repos/refnerf-pytorch/
cd ${DIR}

DEG_VIEW=5
RENDER_CHUNK_SIZE=8192

python3 eval.py \
  --gin_configs="${DIR}/logs/$NAME/$EXP/config.gin" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  --logtostderr

conda deactivate
