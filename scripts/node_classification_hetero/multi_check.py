import os
from check import check
from types import SimpleNamespace

def run(path):
    print(f"Running on {path}")
    results = []
    for idx in range(0, 10):
        args = {
            "path": os.path.join(path, str(idx)),
            "report": "",
            "rerun": 0,
        }
        args = SimpleNamespace(**args)
        print(args)
        result = check(args)
        results.append(result)
    
    # calculate the mean and std
    import numpy as np
    mean = np.mean(results)
    std = np.std(results)
    print(f"Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    run(path)