import numpy as np
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def transform(g):
    # g = dgl.to_bidirected(g)
    g.ndata["feat"] = torch.nn.functional.one_hot(g.ndata["feat"].long(), num_classes=28).float()
    g.edata["feat"] = torch.nn.functional.one_hot(g.edata["feat"].long(), num_classes=4).float()
    return g

def get_graphs(batch_size):
    from dgl.data import ZINCDataset

    data_train = ZINCDataset(mode="train", transform=transform)
    data_valid = ZINCDataset(mode="valid", transform=transform)
    data_test = ZINCDataset(mode="test", transform=transform)

    batch_size = batch_size if batch_size > 0 else len(data_train)
    for g, y in data_train:
        print(g.ndata["feat"].shape)
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
    data_train, data_valid, data_test = get_graphs(args.batch_size)
    g, y = next(iter(data_train))


    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=1,
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
        edge_features=g.edata["feat"].shape[-1],
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
    y_vl = y_vl.unsqueeze(-1).float()
    y_te = y_te.unsqueeze(-1).float()

    rmse_vl_min, rmse_te_min = np.inf, np.inf
        
    for idx in range(args.n_epochs):
        for g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            y = y.unsqueeze(-1).float()
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["feat"].float(), e=g.edata["feat"])
            h = h.mean(0)
            loss = loss + torch.nn.functional.mse_loss(h, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                h_vl, _ = model(g_vl, g_vl.ndata["feat"].float(), e=g_vl.edata["feat"].float())
                h_vl = h_vl.mean(0)
                rmse_vl = torch.sqrt(torch.nn.functional.mse_loss(h_vl, y_vl)).item()
                if early_stopping([rmse_vl]):
                    break

                h_te, _ = model(g_te, g_te.ndata["feat"].float(), e=g_te.edata["feat"].float())
                h_te = h_te.mean(0)
                rmse_te = torch.sqrt(torch.nn.functional.mse_loss(h_te, y_te)).item()

                # print(rmse_vl, rmse_te)

                if rmse_vl < rmse_vl_min:
                    rmse_vl_min = rmse_vl
                    rmse_te_min = rmse_te

    print(rmse_vl_min, rmse_te_min, flush=True)
    return rmse_vl_min, rmse_te_min

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=50)
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)
