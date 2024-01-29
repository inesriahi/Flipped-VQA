#!/bin/bash
#SBATCH --job-name=Fl-all-exp
#SBATCH --account=project_462000189
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --output=logs/trained_all/%A_%a_output.txt
#SBATCH --error=logs/trained_all/%A_%a_error.txt

#SBATCH --array=2-31

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base
# Read parameters from the specific line of the params file
line_number=$SLURM_ARRAY_TASK_ID
read -r audio audio_only audio_merge model_size blr dataset max_seq_len <<< $(awk -v var="$line_number" 'NR==var {print $1, $2, $3, $4, $5, $6, $7}' params.txt)

# Call the main training script with the parameters
./train_script.sh $audio $audio_only $audio_merge $model_size $blr $dataset $max_seq_len $line_number
