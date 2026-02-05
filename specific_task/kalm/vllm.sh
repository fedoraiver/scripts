#!/usr/bin/env bash
source ~/tdy.bashrc

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
runbg vllm-server-8192 vllm serve /mnt/nvme0/tdy/my_models/HY/ --served-model-name HY --port 8001 --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 131072 --gpu-memory-utilization 0.88
sleep 10
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
runbg vllm-server-8192 vllm serve /mnt/nvme0/tdy/my_models/HY/ --served-model-name HY --port 8002 --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 131072 --gpu-memory-utilization 0.88