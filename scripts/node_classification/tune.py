import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.ax import AxSearch
from ray.tune.search import Repeater
import torch
num_gpus = torch.cuda.device_count()
ray.init(num_cpus=num_gpus, num_gpus=num_gpus)

print(num_gpus)

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    acc_vl, acc_te = run(args)
    ray.train.report(dict(acc_vl=acc_vl, acc_te=acc_te))

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S") + "_" + args.data
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(32, 64),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-2),
        "length": tune.randint(3, 16),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_layers": 1, # tune.randint(1, 3),
        "num_samples": 8,
        "n_epochs": 2000,  
        "patience": 100,
        "self_supervise_weight": tune.loguniform(1e-4, 1.0),
        "consistency_weight": tune.loguniform(1e-4, 1.0),
        "dropout": tune.uniform(0.0, 0.5),
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        "split_index": args.split_index,
    }

    tune_config = tune.TuneConfig(
        metric="acc_vl",
        mode="max",
        search_alg=Repeater(AxSearch(), 1),
        num_samples=9000,
    )

    if args.split_index < 0:
        storage_path = os.path.join(os.getcwd(), args.data)
    else:
        storage_path = os.path.join(os.getcwd(), args.data, str(args.split_index))
    
    run_config = air.RunConfig(
        name=name,
        storage_path=storage_path,
        verbose=0,
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--split_index", type=int, default=-1)
    args = parser.parse_args()
    experiment(args)
