import torch
import dgl
from dgl.data import DGLDataset

def process(g, length):
    # g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.remove_self_loop(g)
    g = dgl.reverse(g, copy_ndata=True)
    node = g.ndata["mask"].nonzero().item()
    number_of_walks = 2 ** length
    from dgl.sampling import random_walk
    walks = torch.zeros(0, length + 1, dtype=torch.int)
    while walks.shape[0] < number_of_walks:
        _walks, _ = random_walk(g, [node] * number_of_walks, length=length)
        walks = torch.cat([walks, _walks])
        walks = torch.unique(walks, dim=0)
    # walks = walks.flip(-1)
    # g.ndata["h"] = torch.cat(
    #     [
    #         torch.nn.functional.one_hot(g.ndata["h"][:, 0]).float(),
    #         torch.nn.functional.one_hot(g.ndata["h"][:, 1]).float(),
    #     ],
    #     dim=-1,
    # )
    h = g.ndata["h"][walks]
    return h

class SmallRUMModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=in_features,
            hidden_size=hidden_features,
            batch_first=True,
        )

        self.linear = torch.nn.Linear(
            in_features=hidden_features,
            out_features=out_features,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_repeat = x.shape[1]
        x = x.flatten(0, 1)
        _, h = self.gru(x)
        h = h.view(batch_size, num_repeat, -1)
        h = torch.nn.functional.silu(h)
        h = self.linear(h)
        return h

# class SmallRUMModel(torch.nn.Module):
#     def __init__(self, in_features, out_features, hidden_features):
#         super().__init__()

#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(
#                 in_features=in_features,
#                 out_features=hidden_features,
#             ),
#             torch.nn.SiLU(),
#             torch.nn.Linear(
#                 in_features=hidden_features,
#                 out_features=hidden_features,
#             ),
#             torch.nn.SiLU(),
#             torch.nn.Linear(
#                 in_features=hidden_features,
#                 out_features=out_features,
#             ),
#         )

#     def forward(self, x):
#         x = self.fc(x.flatten(-2, -1))
#         return x

def run(args):
    print(args, flush=True)
    graphs, labels = dgl.load_graphs(args.data)
    x = torch.stack([process(g, length=args.length) for g in graphs], dim=0)
    in_features = (x[..., 0].max()+1, x[..., 1].max()+1)
    y = labels["y"]
    dataset = torch.utils.data.TensorDataset(x, y)
    if args.batch_size == -1:
        batch_size = x.shape[0]
    else:
        batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    model = SmallRUMModel(
        in_features=(in_features[0] + in_features[1]),
        out_features=y.max().item() + 1,
        hidden_features=args.hidden_features,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1000,
        verbose=True,
    )

    for idx in range(args.n_epochs):
        acc = []
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.to("cuda")
                y = y.to("cuda")

            x = torch.cat(
                [
                    torch.nn.functional.one_hot(x[..., 0], in_features[0]).float(),
                    torch.nn.functional.one_hot(x[..., 1], in_features[1]).float(),
                ],
                dim=-1,
            )
            model.train()
            optimizer.zero_grad()
            logits = model(x).mean(-2)
            loss = torch.nn.CrossEntropyLoss()(logits, y)
            acc.append((logits.argmax(dim=-1) == y).float().mean().item())
            loss.backward()
            optimizer.step()
        acc = sum(acc) / len(acc)
        scheduler.step(acc)
        print(acc, flush=True)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/scratch/yw8052/bottleneck/data/5.bin")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=10000000)
    # parser.add_argument("--factor", type=float, default=0.5)
    # parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--self_supervise_weight", type=float, default=10.0)
    parser.add_argument("--consistency_weight", type=float, default=1)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    run(args)