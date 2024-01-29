import torch
import dgl
from datasets import load_dataset
from dgl.dataloading import DataLoader


def get_graph(data):
    src, dst = data["edge_index"]
    src, dst = torch.tensor(src), torch.tensor(dst)
    num_nodes = data["num_nodes"]
    g = dgl.graph(
        data=(src, dst),
        num_nodes=num_nodes,
    )
    y = data["y"]
    y = torch.tensor(y)
    g.ndata["h"] = torch.zeros(num_nodes, 1)
    return g, y


def run(args):
    data = load_dataset("graphs-datasets/CSL")
    data_train, data_valid, data_test = data["train"], data["val"], data["test"]
    data_train = [get_graph(g) for g in data_train]
    data_valid = [get_graph(g) for g in data_valid]
    data_test = [get_graph(g) for g in data_test]
    g_tr = dgl.batch([g for g, _ in data_train])
    y_tr = torch.cat([y for _, y in data_train])
    g_vl = dgl.batch([g for g, _ in data_valid])
    y_vl = torch.cat([y for _, y in data_valid])
    g_te = dgl.batch([g for g, _ in data_test])
    y_te = torch.cat([y for _, y in data_test])

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g_tr.ndata["h"].shape[-1],
        out_features=y_tr.max().int().item()+1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        dropout=args.dropout,
        num_layers=1,
        self_supervise_weight=args.self_supervise_weight,
        activation=getattr(torch.nn, args.activation)(),
        binary=False,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g_tr = g_tr.to("cuda")
        g_vl = g_vl.to("cuda")
        g_te = g_te.to("cuda")
        y_tr = y_tr.to("cuda")
        y_vl = y_vl.to("cuda")
        y_te = y_te.to("cuda")

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for idx in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        h_tr, loss = model(g_tr, g_tr.ndata["h"])
        print(h_tr)
        loss = loss + torch.nn.CrossEntropyLoss()(h_tr, y_tr)
        loss.backward()
        optimizer.step()

        # model.eval()
        # with torch.no_grad():
        #     h_vl, loss = model(g_vl, g_vl.ndata["h"])
        #     accuracy = (h_vl.argmax(dim=-1) == y_vl).float().mean().item()
        #     print(accuracy)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--self_supervise_weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)