#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup torchrun --nproc_per_node=1 train.py > train.log 2>&1 &
