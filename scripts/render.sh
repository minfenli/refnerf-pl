#!/bin/bash

EXP=$1
NAME=$2
DATA_DIR=/media/public_dataset/NeRF/nerf_synthetic/$1

DIR=/media/NFS/fong/refnerf-pl
cd ${DIR}

DEG_VIEW=5
RENDER_CHUNK_SIZE=4096

if [[ "$CONFIG" == *"llff"* ]]; then
  RENDER_PATH=True
else
  RENDER_PATH=False
fi

python3 render.py \
  --gin_configs="${DIR}/exps/logs/${EXP}/${EXP}_${NAME}/config.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DIR}/exps'" \
  --gin_bindings="Config.render_dir = '${DIR}/exps/ckpt/${EXP}_${NAME}/render/'" \
  --gin_bindings="Config.render_path = $RENDER_PATH" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --gin_bindings="Config.render_chunk_size = $RENDER_CHUNK_SIZE" \
  --gin_bindings="NerfMLP.deg_view = $DEG_VIEW" \
  --logtostderr