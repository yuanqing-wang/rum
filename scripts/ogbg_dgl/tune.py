import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.schedulers import ASHAScheduler
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
    run(args)

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S") + "_" + args.data
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(32, 128),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-12, 1e-2),
        "length": tune.randint(3, 8),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": tune.randint(1, 4),
        "num_layers": 1, # tune.randint(1, 3),
        "num_samples": 4,
        "n_epochs": 100,  
        # "patience": 10,
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": 256,
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        # "factor": tune.uniform(0.5, 0.9),
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
    )

    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        search_alg=AxSearch(),
        num_samples=100,
        mode='max',
        metric='acc_vl',
    )

    storage_path = os.path.join(os.getcwd(), args.data)
    
    run_config = air.RunConfig(
        name=name,
        storage_path=storage_path,
        verbose=1,
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="HIV")
    parser.add_argument("--split_index", type=int, default=-1)
    args = parser.parse_args()
    experiment(args)
