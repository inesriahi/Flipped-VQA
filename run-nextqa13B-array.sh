#!/bin/bash
#SBATCH --job-name=FllamaNext13B
#SBATCH --account=project_462000189
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=48:00:00

#SBATCH --output=logs/nextqa/%A_%a_output.txt
#SBATCH --error=logs/nextqa/%A_%a_error.txt

#SBATCH --array 2-6

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base
# pip install fairscale fire sentencepiece transformers timm pandas pysrt

# Get Argument
line_number=$SLURM_ARRAY_TASK_ID

# Read parameters from the specific line of the params file
read -r audio audio_only audio_merge <<< $(awk -v var="$line_number" 'NR==var {print $1, $2, $3}' params.txt)

# Initialize command
cmd="srun torchrun --nproc_per_node 1 train.py --model 13B \
--max_seq_len 128 --batch_size 8 --epochs 15 --warmup_epochs 2 --bias 3.5 --tau 100. --adapter_layer 40 --max_feats 10 --dataset nextqa \
--blr 9e-3 --weight_decay 0.14 --output_dir ./checkpoint/nextqa_13b/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} --accum_iter 2 --vaq --qav --is_generation_task"

# Add flags based on the True/False values
if [ "$audio" = "True" ]; then
    cmd+=" --audio"
fi

if [ "$audio_only" = "True" ]; then
    cmd+=" --audio_only"
fi

# Add audio_merge flag (assuming it's always needed and not a boolean)
cmd+=" --audio_merge $audio_merge"

eval $cmd