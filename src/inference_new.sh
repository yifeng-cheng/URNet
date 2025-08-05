#!/bin/bash

set -x  # 打印命令执行过程

cuda_idx='1,3'
config_path=/media/data/hucao/yifeng/se-cff-test/configs/config.yaml
data_root=/media/data/hucao/yifeng/se-cff-test/data/DSEC
save_root=/media/data/hucao/yifeng/se-cff-test/save_new
checkpoint_path=/media/data/hucao/yifeng/se-cff-test/save_new/weights/final.pth
num_workers=4
NUM_PROC=2  # 启动的 GPU 数量，必须与 cuda_idx 对应

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch \
    --nproc_per_node=$NUM_PROC \
    --master_port=$RANDOM \
    inference_final.py \
    --config_path ${config_path} \
    --data_root ${data_root} \
    --save_root ${save_root} \
    --checkpoint_path ${checkpoint_path} \
    --num_workers ${num_workers}
