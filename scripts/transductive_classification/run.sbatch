#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --mem=5GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

python run.py