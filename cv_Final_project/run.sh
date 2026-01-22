#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# 可选：指定 batch size、epoch 等超参数
BATCH_SIZE=4
EPOCHS=50
LEARNING_RATE=1e-3

# 启动训练
python train.py \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE
