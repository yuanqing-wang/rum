import os
import glob
import json
import dgl
import pandas as pd
import torch
from statistics import mean, stdev

def check(args):
    results = []
    result_paths = glob.glob(args.path + "/*/*/result.json")
    for result_path in result_paths:
        try:
            with open(result_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    result = json.loads(line)
                    results.append(result)
        except:
            pass


    print(len(results), len(result_paths))
    if "__trial_index__" in results[0]["config"]:
        from collections import defaultdict
        config_to_result = defaultdict(list)
        for result in results:
            config = result["config"]
            config.pop("__trial_index__")
            config["training_iteration"] = result["training_iteration"]
            checkpoint = config.pop("checkpoint")
            config_to_result[str(config)].append(
                {"acc_vl": result["acc_vl"], "acc_te": result["acc_te"], "checkpoint": checkpoint}
            )

        results = []
        for config, results_ in config_to_result.items():
            if len(results_) < args.rerun:
                continue
            acc_vl = []
            acc_te = []
            for result in results_: 
                acc_vl.append(result["acc_vl"])
                acc_te.append(result["acc_te"])
            if len(acc_te) == 1:
                acc_te_std = 0
            else:
                acc_te_std = stdev(acc_te)
            acc_vl = mean(acc_vl)
            acc_te = mean(acc_te)    
            results.append({"config": config, "acc_vl": acc_vl, "acc_te": acc_te, "acc_te_std": acc_te_std})
        
    # print(results)
    results = sorted(results, key=lambda x: x["acc_vl"], reverse=True)


    print(results[0])

    if len(args.report) > 1:
        df = pd.DataFrame([result["config"] for result in results])
        df["acc_vl"] = [result["acc_vl"] for result in results]
        df["acc_te"] = [result["acc_te"] for result in results]
        df.to_csv(args.report)
    
    return results[0]["acc_te"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--rerun", type=int, default=3)
    args = parser.parse_args()
    check(args)
