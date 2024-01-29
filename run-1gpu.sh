#!/bin/bash
#SBATCH --job-name=Nextllama2
#SBATCH --account=project_462000189
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=3:00:00

#SBATCH --output=logs/nextqa/%A_output.txt
#SBATCH --error=logs/nextqa/%A_error.txt

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base

jobid=$SLURM_JOB_ID
echo $jobid
srun torchrun --nproc_per_node 1 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 20 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/nextqa/$jobid --accum_iter 2 --debug \
--audio --audio_merge attention --is_generation_task --resume ./checkpoint/nextqa/5627757_6/checkpoint_best.pth --vaq --qav

 #--is_generation_task --resume /scratch/project_462000189/ines/Flipped-VQA/checkpoint/nextqa/5360945/checkpoint_best.pth  
#--audio_only #--audio_merge sum #--is_generation_task --vaq --qav