from turtle import pos
import torch
import numpy as np
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import Subset
from sklearn.model_selection import KFold
from rum.utils import EarlyStopping

def get_graphs(data, batch_size):
    dataset = GINDataset(name=data, self_loop=False)
    dataset.labels = dataset.labels.unsqueeze(-1).float()
    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)
    kf = KFold(n_splits=10)
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in kf.split(idxs):
        train_idxs.append(idxs[train_idx])
        test_idxs.append(idxs[test_idx])
    return dataset, train_idxs, test_idxs
    
def _run(args, data_train, data_valid):
    y_mean = torch.cat([y.float() for g, y in data_train]).float().mean(0)
    g, y = next(iter(data_train))
    pos_weight = (1 - y_mean) / y_mean
    early_stopping = EarlyStopping(patience=args.patience)

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["attr"].shape[-1],
        out_features=1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        dropout=args.dropout,
        self_supervise_weight=args.self_supervise_weight,
        consistency_weight=args.consistency_weight,
        temperature=args.consistency_temperature,
        activation=getattr(torch.nn, args.activation)(),
    )


    if torch.cuda.is_available():
        model = model.cuda()
        pos_weight = pos_weight.cuda()

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    accs = []
        
    for idx in range(args.n_epochs):
        model.train()
        for g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["attr"])
            h = h.mean(0)
            loss = loss + torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
            )(h, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_vl, y_vl = [], []
            model.eval()
            for g, y in data_valid:
                if torch.cuda.is_available():
                    g = g.to("cuda")
                    y = y.to("cuda")
                h, _ = model(g, g.ndata["attr"])
                h_vl.append(h.mean(0))
                y_vl.append(y)

            h_vl, y_vl = torch.cat(h_vl), torch.cat(y_vl)
            acc = ((h_vl.sigmoid() - y_vl).abs() < 0.5).float().mean().item()

            accs.append(acc)
            if early_stopping([-acc]):
                break
    accs = np.array(accs)
    if len(accs) < args.n_epochs:
        accs = np.pad(accs, (0, args.n_epochs-len(accs)), mode="constant", constant_values=accs[-1])
    return accs

def run(args):
    dataset, train_idxs, test_idxs = get_graphs(args.data, args.batch_size)
    accs = []
    for _train_idxs, _test_idxs in zip(train_idxs, test_idxs):
        if args.batch_size < 0:
            batch_size = len(_train_idxs)
        else:
            batch_size = args.batch_size
        data_train = GraphDataLoader(
            Subset(dataset, _train_idxs.tolist()),
            batch_size=batch_size,
            shuffle=True,
        )
        data_valid = GraphDataLoader(
            Subset(dataset, _test_idxs.tolist()),
            batch_size=len(_test_idxs),
            shuffle=False,
        )

        acc_te = _run(args, data_train, data_valid)
        accs.append(acc_te)
    accs = np.stack(accs)
    accs_std = np.std(accs, axis=0)
    accs = np.mean(accs, axis=0)
    accs, accs_std = np.max(accs), accs_std[np.argmax(accs)]
    return accs, accs_std

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MUTAG")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=200)
    # parser.add_argument("--factor", type=float, default=0.5)
    # parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--self_supervise_weight", type=float, default=1e-3)
    parser.add_argument("--consistency_weight", type=float, default=1e-3)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="ELU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)