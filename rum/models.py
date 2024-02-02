from typing import Callable
import torch
import dgl
from dgl.nn import GlobalAttentionPooling
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
            self_supervise_weight=0.05,
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
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight

    def forward(self, g, h, e=None, consistency_weight=None):
        g = g.local_var()
        if consistency_weight is None:
            consistency_weight = self.consistency_weight
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e)
            loss = loss + self.self_supervise_weight * _loss
        h = self.fc_out(h).softmax(-1)
        if self.training:
            _loss = self.consistency(h)
            _loss = _loss * consistency_weight
            loss = loss + _loss
        return h, loss

class RUMGraphRegressionModel(RUMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_out = torch.nn.Sequential(
            self.activation,
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            # torch.nn.BatchNorm1d(self.hidden_features),
            self.activation,
            torch.nn.Dropout(kwargs["dropout"]),
            torch.nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, g, h, e=None):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e)
            loss = loss + self.self_supervise_weight * _loss
        # h = self.activation(h)
        h = h.mean(0)
        g.ndata["h"] = h
        h = dgl.sum_nodes(g, "h")
        h = self.fc_out(h)
        return h, loss