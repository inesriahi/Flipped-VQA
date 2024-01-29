#!/bin/bash
#SBATCH --job-name=audio_features
#SBATCH --account=project_462000189
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00

#SBATCH --output=output_audio_features_%j.txt
#SBATCH --error=errors_audio_features_%j.txt

module use /appl/local/csc/modulefiles/
module load pytorch
# pip install clip
# pip install git+https://github.com/openai/CLIP.git

export PYTHONUSERBASE=/scratch/project_462000189/Jalil/python_base

srun python3 preprocess/extract.py \
    --path ./data/msrvtt/audio \
    --output ./data/msrvtt/features/audio \
    --sample_rate 16000 \
    --targetlength 2240 \
    --num_mel_bins 224 \
    --frame_shift 4.45 \
    --audio_mean -5.889523029327393 \
    --audio_std  3.7667300701141357 \
    --model_dir ./pretrained \


     
#nexqa  
# --audio_mean -6.148953437805176 \
# --audio_std  3.651212215423584 \