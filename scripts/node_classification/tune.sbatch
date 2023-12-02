#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH -p nvidia
#SBATCH --gres=gpu:4

python tune.py # --data CiteseerGraphDataset
