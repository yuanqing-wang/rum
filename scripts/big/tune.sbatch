#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# #SBATCH --partition nvidia

# python tune.py --data FlickrDataset
python tune.py --data RedditDataset
