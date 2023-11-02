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


def objective(config):
    args = SimpleNamespace(**config)
    acc_vl, acc_te = run(args)
    tune.report(acc_vl=acc_vl, acc_te=acc_te)

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.qrandint(8, 128),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "length": tune.randint(4, 16),
        "lr": tune.loguniform(1e-4, 1e-1),
        "temperature": tune.uniform(0.0, 1.0),
        "consistency": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_samples": 16,
        "n_epochs": 1000,
    }

    tune_config = tune.TuneConfig(
        metric="acc_vl",
        mode="max",
        search_alg=HyperOptSearch(),
        num_samples=1000,
    )

    run_config = air.RunConfig(
        name=name,
        storage_path=args.data,
        verbose=1,
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
