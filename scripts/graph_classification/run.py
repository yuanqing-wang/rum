import torch
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader

def get_graphs(data, batch_size):
    dataset = DglGraphPropPredDataset(name=data)
    split_idx = dataset.get_idx_split()
    if batch_size < 0:
        batch_size = len(split_idx["train"])
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=len(split_idx["valid"]), shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=len(split_idx["test"]), shuffle=False, collate_fn=collate_dgl)
    return train_loader, valid_loader, test_loader

def run(args):
    data_train, data_valid, data_test = get_graphs(args.data, args.batch_size)
    g, y = next(iter(data_train))

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["feat"].shape[-1],
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

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    from ogb.graphproppred import Evaluator
    evaluator = Evaluator(name = args.data)
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
    g_vl, y_vl = next(iter(data_valid))
    g_te, y_te = next(iter(data_test))
    if torch.cuda.is_available():
        g_vl = g_vl.to("cuda")
        y_vl = y_vl.to("cuda")
        g_te = g_te.to("cuda")
        y_te = y_te.to("cuda")
    
    acc_vl_max, acc_te_max = 0.0, 0.0
        
    for idx in range(args.n_epochs):
        for g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["feat"].float())
            h = h.mean(0)
            loss = loss + torch.nn.BCEWithLogitsLoss()(h, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                h_vl, _ = model(g_vl, g_vl.ndata["feat"].float())
                h_vl = h_vl.mean(0)
                acc_vl = evaluator.eval({"y_true": y_vl, "y_pred": h_vl})

                if early_stopping([-acc_vl]):
                    break

                h_te, _ = model(g_te, g_te.ndata["feat"].float())
                h_te = h_te.mean(0)
                acc_te = evaluator.eval({"y_true": y_te, "y_pred": h_te})

                if acc_vl > acc_vl_max:
                    acc_vl_max = acc_vl
                    acc_te_max = acc_te

    print(acc_vl_max, acc_te_max, flush=True)
    return acc_vl_max, acc_te_max

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ogbg-molhiv")
    parser.add_argument("--hidden_features", type=int, default=64)
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
    parser.add_argument("--self_supervise_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="ELU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)