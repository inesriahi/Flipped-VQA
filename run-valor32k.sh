#!/bin/bash
#SBATCH --job-name=Fllama2
#SBATCH --account=project_462000189
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=10:00:00

#SBATCH --output=logs/valor32k/%A_output.txt
#SBATCH --error=logs/valor32k/%A_error.txt

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base
# pip install fairscale fire sentencepiece transformers timm pandas pysrt
jobid=$SLURM_JOB_ID
echo $jobid
srun torchrun --nproc_per_node 8 train.py --model 7B \
--max_seq_len 256 --batch_size 8 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset valor32k \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/valor32k/$jobid --accum_iter 2 --is_generation_task \
--vaq --qav # --audio --audio_only --audio_merge sum #--is_generation_task