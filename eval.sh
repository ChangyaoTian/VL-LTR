#!/usr/bin/env bash
set -x

export NCCL_LL_THRESHOLD=0

CONFIG=$1
GPUS=$2
CPUS=$[GPUS*4]
PORT=${PORT:-8886}

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="./checkpoints/eval"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir ${OUTPUT_DIR}
fi

python -u main.py \
    --port=$PORT \
    --num_workers 4 \
    --resume "./checkpoints/${CONFIG_NAME}/checkpoint.pth" \
    --output-dir ${OUTPUT_DIR} \
    --config $CONFIG ${@:3} \
    --eval \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log