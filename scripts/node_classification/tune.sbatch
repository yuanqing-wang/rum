#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition nvidia

# python tune.py --data CoauthorCSDataset 
# python tune.py --data CoauthorPhysicsDataset
# python tune.py --data AmazonCoBuyComputerDataset
# python tune.py --data AmazonCoBuyPhotoDataset

# python tune.py --data CoraGraphDataset
python tune.py --data CiteseerGraphDataset
# python tune.py --data PubmedGraphDataset
