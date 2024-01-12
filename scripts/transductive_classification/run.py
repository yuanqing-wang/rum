import copy
from operator import ne

from requests import get
import numpy as np
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from sklearn.metrics import f1_score
from functools import partial
f1_score = partial(f1_score, average="micro")

def get_graph(data, batch_size):
    from dgl.data import (
        PPIDataset,
    )

    data = getattr(dgl.data, data)
    data_train = data(mode="train")
    data_valid = data(mode="valid")
    data_test = data(mode="test")
    if batch_size < 0:
        batch_size = len(data_train)

    data_train = dgl.dataloading.GraphDataLoader(
        data_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    data_valid = dgl.dataloading.GraphDataLoader(
        data_valid, batch_size=len(data_valid),
    )

    data_test = dgl.dataloading.GraphDataLoader(
        data_test, batch_size=len(data_test),
    )

    
    return data_train, data_valid, data_test

def run(args):
    data_train, data_valid, data_test = get_graph(args.data, args.batch_size)
    g = next(iter(data_train))
    pos_weight = torch.cat([g.ndata["label"].flatten() for g in data_train]).mean().pow(-1)

    from rum.models import RUMModel

    model = RUMModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        temperature=args.consistency_temperature,
        dropout=args.dropout,
        num_layers=1,
        self_supervise_weight=args.self_supervise_weight,
        consistency_weight=args.consistency_weight,
        activation=getattr(torch.nn, args.activation)(),
        final_activation=torch.nn.Identity(),
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda")

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


    # for _ in range(1000):
    #     optimizer.zero_grad()
    #     _, loss = model(g, g.ndata["feat"], consistency_weight=0.0)
    #     loss.backward()
    #     optimizer.step()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode="max",
    #     factor=args.factor,
    #     patience=args.patience,
    # )

    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    acc_vl_max, acc_te_max = 0, 0
    for idx in range(args.n_epochs):
        for g in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda")
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["feat"])
            h = h.mean(0)
            loss = loss + torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                h, 
                g.ndata["label"],
            ) 
            loss.backward()
            optimizer.step()
    
        with torch.no_grad():
            h, _ = model(g, g.ndata["feat"])
            h = h.mean(0)
            acc_tr = f1_score(
                (h.cpu().flatten() > 0) * 1,
                g.ndata["label"].cpu().flatten().int(),
            )

            g = next(iter(data_valid))
            if torch.cuda.is_available():
                g = g.to("cuda")
            h, _ = model(g, g.ndata["feat"])
            h = h.mean(0)
            acc_vl = f1_score(
                (h.cpu().flatten() > 0) * 1,
                g.ndata["label"].cpu().flatten().int(),
            )

            g = next(iter(data_test))
            if torch.cuda.is_available():
                g = g.to("cuda")
            h, _ = model(g, g.ndata["feat"])
            h = h.mean(0)
            acc_te = f1_score(
                (h.cpu().flatten() > 0) * 1,
                g.ndata["label"].cpu().flatten().int(),
            )

            if __name__ == "__main__":
                print(
                    f"Epoch: {idx+1:03d}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Train Acc: {acc_tr:.4f}, "
                    f"Val Acc: {acc_vl:.4f}, "
                    f"Test Acc: {acc_te:.4f}"
                )

            # scheduler.step(acc_vl)

            # if optimizer.param_groups[0]["lr"] < 1e-6:
            #     break

            if acc_vl > acc_vl_max:
                acc_vl_max = acc_vl
                acc_te_max = acc_te
                
            if early_stopping([-acc_vl]):
                break
    
    print(acc_vl_max, acc_te_max, flush=True)
    return acc_vl_max, acc_te_max
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="PPIDataset")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=10000)
    # parser.add_argument("--factor", type=float, default=0.5)
    # parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--self_supervise_weight", type=float, default=1e-5)
    parser.add_argument("--consistency_weight", type=float, default=1)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="ELU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    args = parser.parse_args()
    run(args)
