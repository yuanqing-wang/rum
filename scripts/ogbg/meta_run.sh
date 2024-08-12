for hidden_features in 32 64 128 256; do
    for depth in 2 3 4; do
        for learning_rate in 1e-2 1e-3; do
            echo "hidden_features: $hidden_features, depth: $depth, learning_rate: $learning_rate"
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 23:00 -n 1 \
            python run.py --hidden_features $hidden_features --depth $depth --learning_rate $learning_rate --data ogbg-molpcba
        done
    done
done
