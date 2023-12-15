#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

python tune.py --data CornellDataset --split_index 0
