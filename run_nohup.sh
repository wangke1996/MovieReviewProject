#!/bin/bash
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=6
source activate wk_py36_pytorch
nohup python ./run.py >./logs/$(date "+%Y%m%d_%H%M%S") 2>&1 &
