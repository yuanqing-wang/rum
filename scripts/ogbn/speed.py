from math import degrees
from re import sub
import torch
from ogb import nodeproppred
import ray
from ray import train

def run(args):
    from ogb.nodeproppred import DglNodePropPredDataset

    import os
    import sys
    current_path = os.path.dirname(os.path.realpath(__file__))

    dataset = DglNodePropPredDataset(name=args.dataset, root=os.path.join(current_path, "dataset"))
    split_idx = dataset.get_idx_split()

    g, y = dataset[0]
    g.ndata["train_mask"] = torch.zeros(g.num_nodes(), dtype=torch.bool)
    g.ndata["train_mask"][split_idx["train"]] = True
    g.ndata["val_mask"] = torch.zeros(g.num_nodes(), dtype=torch.bool)
    g.ndata["val_mask"][split_idx["valid"]] = True
    g.ndata["test_mask"] = torch.zeros(g.num_nodes(), dtype=torch.bool)
    g.ndata["test_mask"][split_idx["test"]] = True
    g.ndata["label"] = y.squeeze()

    from rum.models import RUMModel
    model = RUMModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max()+1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        temperature=args.consistency_temperature,
        dropout=args.dropout,
        num_layers=1,
        self_supervise=False,
        consistency_weight=args.consistency_weight,
        degrees=False,
        binary=False,
        activation=getattr(torch.nn, args.activation)(),
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda:0")
    
    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    


    nodes = g.ndata["train_mask"].nonzero().flatten()[torch.randperm(g.ndata["train_mask"].sum())]
    model.eval()
    i = 0
    subsample = nodes[i:i+args.batch_size]
    subsample = subsample.to(g.device)
    h0 = g.ndata["feat"]
    base_memory = torch.cuda.memory_allocated()/(1024**2)
    from tqdm import tqdm
    for _ in range(10):
        for i in tqdm(range(0, g.ndata["train_mask"].sum(), args.batch_size)):
            h, loss = model(g, h0, subsample=subsample)
            current_memory = torch.cuda.memory_allocated()/(1024**2)
            print(f"Memory: {current_memory-base_memory:.2f} MB")
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OGBN Node Property Prediction')
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--hidden_features', type=int, default=64)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--length', type=int, default=2)
    parser.add_argument('--consistency_temperature', type=float, default=1.0)
    parser.add_argument('--self_supervise_weight', type=float, default=1e-3)
    parser.add_argument('--consistency_weight', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='SiLU')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=50)
    args = parser.parse_args()
    run(args)