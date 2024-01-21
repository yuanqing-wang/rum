#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1


python tune.py --data ESOL
# python tune.py --data FreeSolv
python tune.py --data ogbg-molhiv

