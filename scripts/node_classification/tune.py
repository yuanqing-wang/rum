import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.hyperopt import HyperOptSearch
import torch
num_gpus = torch.cuda.device_count()
ray.init(num_cpus=num_gpus, num_gpus=num_gpus)

print(num_gpus)

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    acc_vl, acc_te = run(args)
    session.report(dict(acc_vl=acc_vl, acc_te=acc_te))

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.lograndint(32, 128, base=2),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-2),
        "length": tune.randint(3, 16),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_layers": 1, # tune.randint(1, 3),
        "num_samples": 8,
        "n_epochs": 500,  
        "self_supervise_weight": tune.loguniform(1e-4, 1e-1),
        "consistency_weight": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.0, 0.5),
        "checkpoint": 1,
        "activation": tune.choice(["SiLU", "ELU", "ReLU"]),
    }

    tune_config = tune.TuneConfig(
        metric="_metric/acc_vl",
        mode="max",
        search_alg=HyperOptSearch(),
        num_samples=10000,
    )

    run_config = air.RunConfig(
        name=name,
        storage_path=os.path.join(os.getcwd(), args.data),
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
    args = parser.parse_args()
    experiment(args)
