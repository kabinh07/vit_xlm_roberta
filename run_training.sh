#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIP_CACHE_DIR=/workspace/.cache
nohup torchrun --nproc_per_node=1 train.py > train.log 2>&1 &
