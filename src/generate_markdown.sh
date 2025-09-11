#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0

export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

source $HOME/venvs/grolts_embed/bin/activate

marker ./data/*.pdf --output_dir ./processed_pdfs --workers 10

exit 0
