#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$PYTHONPATH

scene='day'

# 根据 SCENE 设置不同的端口
if [ "$scene" = "day" ]; then
    export CUDA_VISIBLE_DEVICES=0,2
    PORT=29500
elif [ "$scene" = "night" ]; then
    export CUDA_VISIBLE_DEVICES=1,3
    PORT=29501
elif [ "$scene" = "all" ]; then
    export CUDA_VISIBLE_DEVICES=1,3
    PORT=29502
else
    echo "Unknown scene: $scene"
    exit 1
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
PORT=$PORT ./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.440x640.dsec.40k.py 2 $scene ./${scene}_work_dirs