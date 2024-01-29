#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition nvidia

<<<<<<< HEAD
python tune.py --data MUTAG
=======
python tune.py --data NCI1
>>>>>>> 34ab7f1bb61abc76572a21b286ac32e0738bde4a
