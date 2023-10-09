import torch
from .layers import RUMLayer

class RUMModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            **kwargs,
    ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(RUMLayer(hidden_features, hidden_features, **kwargs))

    def forward(self, g, h):
        g = g.local_var()
        h = self.fc_in(h)
        for layer in self.layers:
            h = layer(g, h)
        h = self.fc_out(h)
        return h
    
