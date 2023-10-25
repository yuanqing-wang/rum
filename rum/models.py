from typing import Callable
import torch
from .layers import RUMLayer, Consistency

class RUMModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: Callable = torch.nn.ELU(),
            temperature=0.1,
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
        self.consistency = Consistency(temperature=temperature)

    def forward(self, g, h):
        g = g.local_var()
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(g, h)
            loss = loss + _loss
        h = self.fc_out(h).softmax(-1)
        _loss = self.consistency(h)
        loss = loss + _loss
        return h, loss
    
