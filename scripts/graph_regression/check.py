import os
import glob
import json
import pandas as pd
import torch
import dgl

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
                {"rmse_vl": result["rmse_vl"], "rmse_te": result["rmse_te"]}
            )

        results = []
        for config, results_ in config_to_result.items():
            rmse_vl = 0
            rmse_te = 0
            for result in results_:
                rmse_vl += result["rmse_vl"]
                rmse_te += result["rmse_te"]
            rmse_vl /= len(results_)
            rmse_te /= len(results_)
            results.append({"config": config, "rmse_vl": rmse_vl, "rmse_te": rmse_te})
        
    # print(results)
    results = sorted(results, key=lambda x: x["rmse_te"], reverse=False)

    print(results[0]["rmse_vl"], results[0]["rmse_te"])
    print(results[0]["config"], flush=True)

    if len(args.report) > 1:
        df = pd.DataFrame([result["config"] for result in results])
        df["rmse_vl"] = [result["rmse_vl"] for result in results]
        df["rmse_te"] = [result["rmse_te"] for result in results]
        df.to_csv(args.report)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--rerun", type=int, default=0)
    args = parser.parse_args()
    check(args)
