#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIP_CACHE_DIR=/mnt/data/.cache
nohup torchrun --nproc_per_node=1 train.py > train.log 2>&1 &
