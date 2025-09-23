#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0

export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

source $HOME/venvs/grolts_embed/bin/activate

marker ./data/ptsd --output_dir ./processed_pdfs/ptsd --workers 10
marker ./data/achievement --output_dir ./processed_pdfs/achievement --workers 10
marker ./data/delinquency --output_dir ./processed_pdfs/delinquency --workers 10
marker ./data/wellbeing --output_dir ./processed_pdfs/wellbeing --workers 10

exit 0
