#!/bin/bash
#SBATCH --job-name=llama2
#SBATCH --account=project_462000189
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=3:00:00

#SBATCH --output=outputs/output_%A.txt
#SBATCH --error=errors/errors_%A.txt

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base
# pip install fairscale fire sentencepiece transformers timm pandas pysrt

srun torchrun --nproc_per_node 8 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/nextqa --accum_iter 2 --vaq --qav