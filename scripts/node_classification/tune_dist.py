from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.hyperopt import HyperOptSearch
import torch
import os
ray.init(num_cpus=os.cpu_count())
LSF_COMMAND = "bsub -q gpuqueue -gpu " +\
"\"num=1:j_exclusive=yes\" -R \"rusage[mem=5] span[ptile=1]\" -W 0:10 -Is "

PYTHON_COMMAND =\
"python /data/chodera/wangyq/rum/scripts/node_classification/run.py"


def args_to_command(args):
    command = LSF_COMMAND + "\""
    command += PYTHON_COMMAND
    for key, value in args.items():
        command += f" --{key} {value}"
    command += "\""
    return command

def lsf_submit(command):
    import subprocess
    print("--------------------")
    print("Submitting command:")
    print(command)
    print("--------------------")
    output = subprocess.getoutput(command)
    return output

def parse_output(output):
    line = output.split("\n")[-1]
    if "ACCURACY" not in line:
        print(output, flush=True)
        return 0.0, 0.0
    _, accuracy_vl, accuracy_te = line.split(",")
    return float(accuracy_vl), float(accuracy_te)

def objective(args):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    args["checkpoint"] = checkpoint
    command = args_to_command(args)
    output = lsf_submit(command)
    accuracy, accuracy_te = parse_output(output)
    session.report({"accuracy": accuracy, "accuracy_te": accuracy_te})

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(8, 32),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "length": tune.randint(4, 16),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "temperature": tune.uniform(0.0, 1.0),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_layers": tune.randint(1, 3),
        "num_samples": 4,
        "n_epochs": 1000,
        "self_supervise_weight": tune.loguniform(1e-4, 1e-1),
        "consistency_weight": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.0, 1.0),
    }

    tune_config = tune.TuneConfig(
        metric="_metric/accuracy",
        mode="max",
        search_alg=HyperOptSearch(),
        num_samples=1000,
    )

    run_config = air.RunConfig(
        name=name,
        storage_path=args.data,
        # verbose=1,
    )

    tuner = tune.Tuner(
        objective,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    experiment(args)
