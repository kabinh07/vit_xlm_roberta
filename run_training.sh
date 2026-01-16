#!/bin/bash

# Set environment variables for optimal GPU training
# export CUDA_VISIBLE_DEVICES="0,1,3"
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=1

# Use torchrun instead of torch.distributed.launch
# Format: torchrun --nproc_per_node=NUM_GPUS train.py
nohup torchrun --nproc_per_node=3 train.py > train.log 2>&1 &
