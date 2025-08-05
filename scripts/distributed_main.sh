#!/bin/bash

set -x

cuda_idx='0, 3'
config_path=/media/data/hucao/yifeng/se-cff-test/configs/config.yaml
data_root=/media/data/hucao/yifeng/se-cff-test/data/DSEC
save_root=/media/data/hucao/yifeng/se-cff-test/save_new
#checkpoint_path=/media/data/hucao/yifeng/se-cff-main/checkpoint.pth
num_workers=4
NUM_PROC=2

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} #--checkpoint_path ${checkpoint_path}
