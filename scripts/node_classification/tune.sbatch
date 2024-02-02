#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# #SBATCH --partition nvidia

# python tune.py --data CoauthorCSDataset 
# python tune.py --data CoauthorPhysicsDataset
python tune.py --data AmazonCoBuyComputerDataset
# python tune.py --data AmazonCoBuyPhotoDataset

# python tune.py --data CoraGraphDataset
# python tune.py --data CiteseerGraphDataset
# python tune.py --data PubmedGraphDataset

# python tune.py --data CornellDataset --split_index $IDX --directed 1
