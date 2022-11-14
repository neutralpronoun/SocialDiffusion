#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=digress_github
#SBATCH --ntasks-per-node=12
#SBATCH --time=00:00:02

module load cuda

