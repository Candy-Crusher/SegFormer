#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
SCENE=$3
WORK_DIR=$4
PORT=${PORT:-29500}  # 设置默认端口为 29500，如果未指定 PORT 则使用该端口

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG $SCENE $WORK_DIR --launcher pytorch ${@:5}