#!/bin/bash
#SBATCH --time=7:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python run.py
