#!/bin/bash
source activate wk_py36_pytorch
nohup python ./run.py >./logs/$(date "+%Y%m%d_%H%M%S") 2>&1 &
