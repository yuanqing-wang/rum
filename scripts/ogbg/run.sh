#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=20] span[ptile=1]"
#BSUB -W 23:00
#BSUB -n 1

# python run.py --data ogbg-molpcba
python run.py --data ogbg-molhiv

