import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def get_graph(data):
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
        CornellDataset,
    )

    g = locals()[data](verbose=False)[0]
    g = dgl.remove_self_loop(g)

    if "train_mask" not in g.ndata:
        g.ndata["train_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["val_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["test_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)

        train_idxs = torch.tensor([], dtype=torch.int32)
        val_idxs = torch.tensor([], dtype=torch.int32)
        test_idxs = torch.tensor([], dtype=torch.int32)

        n_classes = g.ndata["label"].shape[-1]
        for idx_class in range(n_classes):
            idxs = torch.where(g.ndata["label"][:, idx_class] == 1)[0]
            assert len(idxs) > 50
            idxs = idxs[torch.randperm(len(idxs))]
            _train_idxs = idxs[:20]
            _val_idxs = idxs[20:50]
            _test_idxs = idxs[50:]
            train_idxs = torch.cat([train_idxs, _train_idxs])
            val_idxs = torch.cat([val_idxs, _val_idxs])
            test_idxs = torch.cat([test_idxs, _test_idxs])

        g.ndata["train_mask"][train_idxs] = True
        g.ndata["val_mask"][val_idxs] = True
        g.ndata["test_mask"][test_idxs] = True
    return g

def run(args):
    g = get_graph(args.data)
    from rum.models import RUMModel
    model = RUMModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max()+1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda")

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for idx in range(args.n_epochs):
        optimizer.zero_grad()
        h = model(g, g.ndata["feat"]).softmax(-1)
        loss = torch.nn.NLLLoss()(
            h[g.ndata["train_mask"]], 
            g.ndata["label"][g.ndata["train_mask"]],
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            h = model(g, g.ndata["feat"]).softmax(-1)
            train_acc = (
                h.argmax(-1)[g.ndata["train_mask"]] == g.ndata["label"][g.ndata["train_mask"]]
            ).float().mean()
            val_acc = (
                h.argmax(-1)[g.ndata["val_mask"]] == g.ndata["label"][g.ndata["val_mask"]]
            ).float().mean()
            test_acc = (
                h.argmax(-1)[g.ndata["test_mask"]] == g.ndata["label"][g.ndata["test_mask"]]
            ).float().mean()
            print(
                f"Epoch: {idx+1:03d}, "
                f"Loss: {loss.item():.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}"
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=10000)
    args = parser.parse_args()
    run(args)