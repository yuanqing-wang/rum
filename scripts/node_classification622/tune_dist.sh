#BSUB -o %J.stdout
#BSUB -R "rusage[mem=5/task] span[hosts=1]"
#BSUB -W 23:59
#BSUB -n 8

python tune_dist.py 

