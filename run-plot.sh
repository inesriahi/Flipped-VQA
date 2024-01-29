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

srun python3 plot_learning_curves.py