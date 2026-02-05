#!/bin/bash

nohup torchrun --nproc_per_node=1 train.py > train.log 2>&1 &
