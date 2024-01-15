import os
import glob
import json
import torch
import dgl
import pandas as pd

def check(args):
    results = []
    result_paths = glob.glob(args.path + "/*/*/result.json")
    for result_path in result_paths:
        try:
            with open(result_path, "r") as f:
                result_str = f.read()
                result = json.loads(result_str)
            results.append(result)
        except:
            pass

    if "__trial_index__" in results[0]["config"]:
        from collections import defaultdict
        config_to_result = defaultdict(list)
        for result in results:
            config = result["config"]
            config.pop("__trial_index__")
            config.pop("checkpoint")
            config_to_result[str(config)].append(
                {"acc_vl": result["acc_vl"], "acc_te": result["acc_te"]}
            )

        results = []
        for config, results_ in config_to_result.items():
            acc_vl = 0
            acc_te = 0
            for result in results_:
                acc_vl += result["acc_vl"]
                acc_te += result["acc_te"]
            acc_vl /= len(results_)
            acc_te /= len(results_)
            results.append({"config": config, "acc_vl": acc_vl, "acc_te": acc_te})
        
    # print(results)
    results = sorted(results, key=lambda x: x["acc_vl"], reverse=True)


    print(results[0])
    print(results[0]["acc_vl"], results[0]["acc_te"])
    print(results[0]["config"], flush=True)

    if len(args.report) > 1:
        df = pd.DataFrame([result["config"] for result in results])
        df["acc_vl"] = [result["acc_vl"] for result in results]
        df["acc_te"] = [result["acc_te"] for result in results]
        df.to_csv(args.report)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--rerun", type=int, default=0)
    args = parser.parse_args()
    check(args)
