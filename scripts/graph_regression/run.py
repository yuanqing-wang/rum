import numpy as np
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def get_graphs(data, batch_size):
    from dgllife.data import (
        ESOL,
        FreeSolv,
        Lipophilicity,
    )
    from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
    )
    data = locals()[data](
        node_featurizer=CanonicalAtomFeaturizer("h0"),
        edge_featurizer=CanonicalBondFeaturizer("e0"),
    )
    from dgllife.utils import RandomSplitter
    splitter = RandomSplitter()
    data_train, data_valid, data_test = splitter.train_val_test_split(
        data, frac_train=0.8, frac_val=0.1, frac_test=0.1, 
        # random_state=args.seed,
    )

    _, g, y = next(iter(dgl.dataloading.GraphDataLoader(
        data_train, batch_size=len(data_train),
    )))

    batch_size = batch_size if batch_size > 0 else len(data_train)

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
    data_train, data_valid, data_test = get_graphs(args.data, args.batch_size)
    _, g, y = next(iter(data_train))

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["h0"].shape[-1],
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
    _, g_vl, y_vl = next(iter(data_valid))
    _, g_te, y_te = next(iter(data_test))
    if torch.cuda.is_available():
        g_vl = g_vl.to("cuda")
        y_vl = y_vl.to("cuda")
        g_te = g_te.to("cuda")
        y_te = y_te.to("cuda")
    
    rmse_vl_min, rmse_te_min = np.inf, np.inf
        
    for idx in range(args.n_epochs):
        for _, g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["h0"])
            h = h.mean(0)
            loss = loss + torch.nn.functional.mse_loss(h, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                h_vl, _ = model(g_vl, g_vl.ndata["h0"])
                h_vl = h_vl.mean(0)
                rmse_vl = torch.sqrt(torch.nn.functional.mse_loss(h_vl, y_vl)).item()
                if early_stopping([rmse_vl]):
                    break

                h_te, _ = model(g_te, g_te.ndata["h0"])
                h_te = h_te.mean(0)
                rmse_te = torch.sqrt(torch.nn.functional.mse_loss(h_te, y_te)).item()

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
