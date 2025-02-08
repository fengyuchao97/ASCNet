#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoints
data_name=GZ

img_size=256
batch_size=8
lr=0.01
max_epochs=200
net_G=ASCNet
lr_policy=linear

split=train
split_val=val
project_name=ASCNet-GZ

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}