from re import split
import numpy as np
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from ray import train
import os
from pathlib import Path

def get_graphs(data, batch_size):
    global pos_weight
    from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
    from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    from dgllife.data import HIV, PCBA
    from dgl.dataloading import GraphDataLoader
    from dgl.data.utils import Subset

    featurizer = SMILESToBigraph(
        node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat'),
        edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e'),
    )
    dataset = locals()[args.data](
        featurizer, 
        load=True,
        cache_file_path=os.path.join(os.path.dirname(__file__), f"{data.lower()}_dglgraph.bin"),
    )

    evaluator = Evaluator(name=f"ogbg-mol{data.lower()}")
    split_idx = DglGraphPropPredDataset(name=f"ogbg-mol{data.lower()}").get_idx_split()
    pos_weight = dataset.task_pos_weights(split_idx["train"])[0]
    data_train = [(dataset[idx][1], dataset[idx][2]) for idx in split_idx["train"]]
    data_valid = [(dataset[idx][1], dataset[idx][2]) for idx in split_idx["valid"]]
    data_test = [(dataset[idx][1], dataset[idx][2]) for idx in split_idx["test"]]

    batch_size = batch_size if batch_size > 0 else len(data_train)
    data_train = GraphDataLoader(
        data_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_dgl,
    )

    data_valid = GraphDataLoader(
        data_valid, batch_size=batch_size, collate_fn=collate_dgl,
    )

    data_test = GraphDataLoader(
        data_test, batch_size=batch_size, collate_fn=collate_dgl,
    )

    return data_train, data_valid, data_test, evaluator

def run(args):
    print(args, flush=True)
    metric = {
        "HIV": "rocauc",
        "PCBA": "ap",
    }[args.data]
    data_train, data_valid, data_test, evaluator = get_graphs(args.data, args.batch_size)

    y = torch.cat([y for _, y in data_train])
    g, y = next(iter(data_train))
    print(g)
    h = g.ndata["feat"]

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=h.shape[-1],
        out_features=y.shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        dropout=args.dropout,
        num_layers=1,
        self_supervise=False,
        activation=getattr(torch.nn, args.activation)(),
        binary=True,
        degrees=False,
        edge_features=g.edata["e"].shape[-1],
    )
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max",
        factor=args.factor,
        patience=args.patience,
    )

    acc_vl_max, acc_te_max = 0.0, 0.0
        
    for idx in range(args.n_epochs):
        for g, y in data_train:
            model.train()
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            h = g.ndata["feat"]
            h, loss = model(g, h, e=g.edata["e"])
            if y.isnan().any():
                h = h[~y.isnan()]
                y = y[~y.isnan()]
            loss = loss + torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(h, y.float())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            h_vl = []
            y_vl = []
            h_te = []
            y_te = []
            for g, y in data_valid:
                if torch.cuda.is_available():
                    g = g.to("cuda")
                    y = y.to("cuda")
                h = g.ndata["feat"]
                h, _ = model(g, h, e=g.edata["e"])
                h_vl.append(h.detach())
                y_vl.append(y)

            for g, y in data_test:
                if torch.cuda.is_available():
                    g = g.to("cuda")
                    y = y.to("cuda")
                h = g.ndata["feat"]
                h, _ = model(g, h, e=g.edata["e"])
                h_te.append(h.detach())
                y_te.append(y)

            h_vl = torch.cat(h_vl, dim=0)
            y_vl = torch.cat(y_vl, dim=0)
            h_te = torch.cat(h_te, dim=0)
            y_te = torch.cat(y_te, dim=0)
            # if y_vl.isnan().any():
            #     h_vl = h_vl[~y_vl.isnan()]
            #     y_vl = y_vl[~y_vl.isnan()]
            #     h_te = h_te[~y_te.isnan()]
            #     y_te = y_te[~y_te.isnan()]

            #     h_te = h_te.unsqueeze(-1)
            #     y_te = y_te.unsqueeze(-1)
            #     h_vl = h_vl.unsqueeze(-1)
            #     y_vl = y_vl.unsqueeze(-1)
            acc_vl = evaluator.eval({"y_true": y_vl.cpu(), "y_pred": h_vl.cpu()})[metric]
            acc_te = evaluator.eval({"y_true": y_te.cpu(), "y_pred": h_te.cpu()})[metric]

            scheduler.step(acc_vl)
            if __name__ == "__main__":
                print(acc_vl, acc_te, flush=True)
            else:
                train.report(dict(acc_vl=acc_vl, acc_te=acc_te))

            if acc_vl > acc_vl_max:
                acc_vl_max = acc_vl
                acc_te_max = acc_te

    return acc_vl_max, acc_te_max

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="HIV")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--self_supervise_weight", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=100)
    args = parser.parse_args()
    run(args)
