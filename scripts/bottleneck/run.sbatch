#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem=10GB
#SBATCH --job-name=bottle
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

python run.py --data /scratch/yw8052/bottleneck/data/7.bin --length 7
# python run.py --data /scratch/yw8052/bottleneck/data/6.bin --length 6
# python run.py --data /scratch/yw8052/bottleneck/data/5.bin --length 5

