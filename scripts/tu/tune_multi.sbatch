#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition nvidia

python tune_multi.py --data COLLAB
