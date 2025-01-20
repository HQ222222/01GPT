#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# export PYTHONPATH=$PYTHONPATH:$(pwd)

nohup torchrun --nproc_per_node 2 pretrainer.py > /data2/minigpt/models/20241210/20250114.log 2>&1 &

