#!/bin/bash

# Get parameters passed from the wrapper script
audio=$1
audio_only=$2
audio_merge=$3
model_size=$4
blr=$5
dataset=$6
max_seq_len=$7
line_number=$8

# Your training command
cmd="srun torchrun --nproc_per_node 1 train.py --model $model_size \
--max_seq_len $max_seq_len --batch_size 8 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --adapter_layer 40 --max_feats 10 --dataset $dataset \
--blr $blr --weight_decay 0.14 --output_dir ./checkpoint/${dataset}_${model_size}/${SLURM_JOB_ID}_${line_number} --accum_iter 2 --vaq --qav --is_generation_task"

# Add flags based on the True/False values
if [ "$audio" = "True" ]; then
    cmd+=" --audio"
fi

if [ "$audio_only" = "True" ]; then
    cmd+=" --audio_only"
fi

cmd+=" --audio_merge $audio_merge"

# Redirect output and error to the specified paths
eval $cmd
