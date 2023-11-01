from typing import Callable, Self
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
            self_superwise_weight=0.05,
            consistency_weight=0.01,
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
            self.layers.append(RUMLayer(hidden_features, hidden_features, in_features, **kwargs))
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_superwise_weight = self_superwise_weight
        self.consistency_weight = consistency_weight

    def forward(self, g, h):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(g, h, h0)
            loss = loss + self.self_superwise_weight * _loss
        h = self.fc_out(h).softmax(-1)
        _loss = self.consistency(h)
        _loss = _loss * self.consistency_weight
        loss = loss + _loss
        return h, loss

    def train_self_supervised(self, g, h):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer.train_self_supervised(g, h, h0)
            loss = loss + _loss
        return h, loss
    
