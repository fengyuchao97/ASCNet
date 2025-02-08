#!/usr/bin/env bash

gpus=0

data_name=GZ
net_G=ASCNet
split=test
project_name=ASCNet-GZ
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


