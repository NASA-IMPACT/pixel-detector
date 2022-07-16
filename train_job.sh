#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --account=training2206
#SBATCH --output=output.out
#SBATCH --error=error.er
#SBATCH --time=2:00:00
#SBATCH --job-name=TESTTENSORFLOW
#SBATCH --gres=gpu:4 --partition=dc-gpu
#SBATCH --hint=nomultithread

ml Stages/2022
ml CUDA/11.5
ml cuDNN/8.3.1.22-CUDA-11.5

export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "Starting training"

source /p/project/training2206/<username>/pixel-detector/.venv/bin/activate
srun python code/train.py

echo "DONE"
