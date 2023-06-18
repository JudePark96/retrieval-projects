#!/bin/bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3;

CORES=$(lscpu | grep Core | awk '{print $4}')
SOCKETS=$(lscpu | grep Socket | awk '{print $2}')
TOTAL_CORES=`expr $CORES \* $SOCKETS`

#KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
#KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
#export $KMP_SETTING
#export KMP_BLOCKTIME=$KMP_BLOCKTIME

python3 train.py