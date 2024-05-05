#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=5] span[ptile=1]"
#BSUB -W 12:00
#BSUB -n 1

# python tune.py --data CornellDataset --split_index 0
# python tune.py --data TexasDataset --split_index 0
python tune.py --data WisconsinDataset --split_index 0

