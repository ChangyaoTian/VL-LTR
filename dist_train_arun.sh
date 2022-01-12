#!/usr/bin/env bash
set -x

export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

PARTITION=$1
CONFIG=$2
GPUS=$3
CPUS=$[GPUS*4]
PORT=${PORT:-6666}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="./checkpoints/${CONFIG_NAME}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir ${OUTPUT_DIR}
fi

srun -p $PARTITION \
    --quotatype=auto \
    --job-name=${CONFIG_NAME} \
    -n ${GPUS} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task 4 \
    python -u main.py \
    --port=$PORT \
    --num_workers 4 \
    --config $CONFIG ${@:4} \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log
