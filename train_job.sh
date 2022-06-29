#!/usr/bin/env bash

# Slurm job configuration

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --account=training2206
#SBATCH --output=output.out
#SBATCH --error=error.er
#SBATCH --time=2:00:00
#SBATCH --job-name=TESTTENSORFLOW
#SBATCH --gres=gpu:1 --partition=gpus
#SBATCH --hint=nomultithread

ml CUDA/11.5
ml cuDNN/8.3.1.22-CUDA-11.5
echo "Starting training"
source /p/project/training2206/<username>/pixel-detector-feature-wmts_input/.env_jusuf/bin/activate
srun python code/train.py
echo "DONE"
