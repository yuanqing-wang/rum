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
                lines = f.readlines()
                for line in lines:
                    result = json.loads(line)
            results.append(result)
        except:
            pass
    print(results)
    # print(results)
    results = sorted(results, key=lambda x: x["acc_vl"], reverse=True)

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
