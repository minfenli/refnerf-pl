#!/bin/bash

NAME=$1
EXP=$2
CONFIG=$3
DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refnerf

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"

DIR=/users/visics/gkouros/projects/nerf-repos/refnerf-pytorch/
cd ${DIR}

DEG_VIEW=5
BATCH_SIZE=1024
RENDER_CHUNK_SIZE=1024
MAX_STEPS=1000

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

CONFIG_PATH="$CONFIG"

#TODO: DELETEME
rm -rf ${DIR}/logs/$NAME/$EXP

python3 train.py \
  --gin_configs="$CONFIG_PATH" \
  --gin_bindings="Config.max_steps = $MAX_STEPS" \
  --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW"
  # && \
  # python3 render.py \
  #   --gin_configs="${DIR}/logs/$NAME/$EXP/config.gin" \
  #   --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  #   --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  #   --gin_bindings="Config.render_dir = '${DIR}/logs/$NAME/$EXP/render/'" \
  #   --gin_bindings="Config.render_path = $RENDER_PATH" \
  #   --gin_bindings="Config.render_path_frames = 480" \
  #   --gin_bindings="Config.render_video_fps = 60" \
  #   --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  #   --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  #   --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  # && \
  # python3 eval.py \
  # --gin_configs="${DIR}/logs/$NAME/$EXP/config.gin" \
  # --gin_bindings="Config.data_dir = '${DIR}/data/$NAME'" \
  # --gin_bindings="Config.checkpoint_dir = '${DIR}/logs/$NAME/$EXP'" \
  # --gin_bindings="Config.batch_size = $BATCH_SIZE" \
  # --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  # --gin_bindings="NerfMLP.deg_view = $DEG_VIEW"

conda deactivate
