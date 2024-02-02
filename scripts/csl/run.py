import sched
import torch
import dgl
from datasets import load_dataset
from dgl.dataloading import DataLoader
from rum.random_walk import uniform_random_walk, uniqueness

class SmallRUMModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_samples):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_samples = num_samples
        self.gru = torch.nn.GRU(
            input_size=in_features,
            hidden_size=hidden_features,
            batch_first=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=hidden_features,
                out_features=hidden_features,
            ),
            # torch.nn.LayerNorm(hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=hidden_features,
                out_features=out_features,
            ),
        )

    def forward(self, g):
        walks, eids = uniform_random_walk(
            g=g, 
            num_samples=self.num_samples, 
            length=self.in_features,
        )

        uniqueness_walk = uniqueness(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = torch.nn.functional.one_hot(uniqueness_walk, num_classes=self.in_features).float()
        h0 = torch.zeros(1, self.num_samples * g.number_of_nodes(), self.hidden_features, device=uniqueness_walk.device)
        _, h = self.gru(uniqueness_walk.flatten(0, 1), h0)
        h = h.mean(0)
        h = h.reshape(self.num_samples, g.number_of_nodes(), self.hidden_features)
        h = self.fc(h).mean(0)
        g.ndata["h"] = h
        h = dgl.sum_nodes(g, "h")
        return h

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

    model = SmallRUMModel(
        in_features=args.length,
        out_features=y_tr.max().int().item()+1,
        hidden_features=args.hidden_features,
        num_samples=args.num_samples,
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.8,
        patience=100,
        verbose=True,
    )
    accuracy_vl_max = 0.0
    accuracy_te_max = 0.0
    for idx in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        h_tr = model(g_tr)
        # print(h_tr)
        loss = torch.nn.CrossEntropyLoss()(h_tr, y_tr)
        accuracy_tr = (h_tr.argmax(dim=-1) == y_tr).float().mean().item()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h_vl = model(g_vl)
            accuracy_vl = (h_vl.argmax(dim=-1) == y_vl).float().mean().item()
            h_te = model(g_te)
            accuracy_te = (h_te.argmax(dim=-1) == y_te).float().mean().item()
            scheduler.step(accuracy_vl)
        if accuracy_vl > accuracy_vl_max:
            accuracy_vl_max = accuracy_vl
            accuracy_te_max = accuracy_te
        print(accuracy_tr, accuracy_vl, accuracy_te, accuracy_te_max, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_epochs", type=int, default=20000000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)