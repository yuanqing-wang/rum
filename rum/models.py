from typing import Callable
import torch
from .layers import RUMLayer

class RUMModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: Callable = torch.nn.ELU(),
            **kwargs,
    ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=True)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(RUMLayer(hidden_features, hidden_features, **kwargs))
        self.activation = activation

    def forward(self, g, h):
        g = g.local_var()
        h = self.fc_in(h)
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h = layer(g, h)
        h = self.fc_out(h).softmax(-1)
        return h
    
