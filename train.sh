#!/bin/bash -e

nproc_per_node=${nproc_per_node:-2}
num_workers=${num_workers:-32}
variable='i10fg'
epochs=120
checkpoint_dir='checkpoints'
loss='MaskedCharbonnierLoss'
model='SwinT2UNet' #  'UNet' 'DCNN', 'GoogleUNet', 'SwinT2UNet'
batch_size=16
transform='standard'
train_years_range='2018,2021'
wandb_id='v43gryq7'

torchrun --nproc_per_node=$nproc_per_node train.py \
--variable $variable \
--epochs $epochs \
--checkpoint_dir $checkpoint_dir \
--batch_size $batch_size \
--model $model \
--loss $loss \
--transform $transform \
--train_years_range $train_years_range \
--num_workers $num_workers
--wandb_id $wandb_id
--resume