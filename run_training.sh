#!/bin/bash

nohup torchrun --nproc_per_node=3 train.py > train.log 2>&1 &
