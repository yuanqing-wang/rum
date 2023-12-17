import math
import torch
import dgl
from .random_walk import uniform_random_walk, uniqueness
from .rnn import GRU
from functools import lru_cache

@lru_cache(maxsize=1)
def to_bidirected(g):
    g = g.local_var()
    g = dgl.to_bidirected(g.to("cpu")).to(g.device)
    return g

class RUMLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            original_features: int,
            num_samples: int,
            length: int,
            dropout: float = 0.2,
            rnn: torch.nn.Module = GRU,
            random_walk: callable = uniform_random_walk,
            activation: callable = torch.nn.Identity(),
            **kwargs,
    ):
        super().__init__()
        # out_features = out_features // 2
        self.rnn = rnn(in_features + 2 * out_features + 1, out_features, **kwargs)
        self.rnn_walk = rnn(2, out_features, bidirectional=True, **kwargs)
        self.out_features = out_features
        # self.fc = torch.nn.Linear(length, length, bias=True)
        # self.fc_self = torch.nn.Linear(in_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.random_walk = random_walk
        self.num_samples = num_samples
        self.length = length
        self.dropout = torch.nn.Dropout(dropout)
        self.self_supervise = SelfSupervise(in_features, original_features)
        self.activation = activation

    def forward(self, g, h, y0):
        """Forward pass.

        Parameters
        ----------
        g : DGLGraph
            The graph.

        h : Tensor
            The input features.

        Returns
        -------
        h : Tensor
            The output features.
        """
        walks = self.random_walk(
            g=g, 
            num_samples=self.num_samples, 
            length=self.length,
        )
        uniqueness_walk = uniqueness(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = uniqueness_walk / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * math.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )
        h = h[walks]
        degrees = g.in_degrees(walks.flatten()).float().reshape(*walks.shape).unsqueeze(-1)
        degrees = degrees / degrees.max()
        num_directions = 2 if self.rnn_walk.bidirectional else 1
        h0 = torch.zeros(self.rnn_walk.num_layers * num_directions, *h.shape[:-2], self.out_features, device=h.device)
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0)
        h_walk = h_walk.mean(0, keepdim=True)
        h = torch.cat([h, y_walk, degrees], dim=-1)
        y, h = self.rnn(h, h_walk)
        if self.training:
            loss = self.self_supervise(y, y0[walks])
        else:
            loss = 0.0
        h = self.activation(h)
        h = h.mean(0)
        h = self.dropout(h)
        return h, loss

class RUMDirectedLayer(RUMLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn_walk = type(self.rnn_walk)(3, self.out_features, bidirectional=True)

    def forward(self, g, h, y0):
        """Forward pass.

        Parameters
        ----------
        g : DGLGraph
            The graph.

        h : Tensor
            The input features.

        Returns
        -------
        h : Tensor
            The output features.
        """
        g = g.local_var()
        g_ = to_bidirected(g)
        walks = self.random_walk(
            g=g_, 
            num_samples=self.num_samples, 
            length=self.length,
        )

        uniqueness_walk = uniqueness(walks)

        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = uniqueness_walk / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * math.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )

        src, dst = walks[..., :-1], walks[..., 1:]
        has_fwd_edge = g.has_edges_between(src.flatten(), dst.flatten()).reshape(*src.shape) * 1.0
        has_bwd_edge = g.has_edges_between(dst.flatten(), src.flatten()).reshape(*src.shape) * 1.0
        has_edge = has_fwd_edge - has_bwd_edge
        has_edge = has_edge.unsqueeze(-1)

        walk_embedding = torch.zeros(
            uniqueness_walk.shape[:-2] 
            + (uniqueness_walk.shape[-2] 
            + has_edge.shape[-2],) + (3,), 
            device=uniqueness_walk.device,
        )

        walk_embedding[..., ::2, :2] = uniqueness_walk
        walk_embedding[..., 1::2, 2:3] = has_edge

        h = h[walks]
        degrees = g.in_degrees(walks.flatten()).float().reshape(*walks.shape).unsqueeze(-1)
        degrees = degrees / degrees.max()
        num_directions = 2 if self.rnn_walk.bidirectional else 1
        h0 = torch.zeros(self.rnn_walk.num_layers * num_directions, *h.shape[:-2], self.out_features, device=h.device)
        y_walk, h_walk = self.rnn_walk(walk_embedding, h0)
        # y_walk = y_walk[..., ::2, :]
        h_walk = h_walk.mean(0, keepdim=True)
        
        y_walk = torch.cat(
            [
                y_walk,
                torch.zeros(*y_walk.shape[:-1], h.shape[-1] + 1, device=y_walk.device),
            ],
            dim=-1,
        )

        y_walk[..., ::2, 2*self.out_features:-1] = h
        y_walk[..., ::2, -1:] = degrees
        # h = torch.cat([h, y_walk, degrees], dim=-1)
        y, h = self.rnn(y_walk, h_walk)
        y = y[..., ::2, :]
        if self.training:
            loss = self.self_supervise(y, y0[walks])
        else:
            loss = 0.0
        h = self.activation(h)
        h = h.mean(0)
        h = self.dropout(h)
        return h, loss    
    
class Consistency(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, probs):
        avg_probs = probs.mean(0)
        sharpened_probs = avg_probs.pow(1 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdim=True)
        loss = (sharpened_probs - avg_probs).pow(2).sum(-1).mean()
        return loss

class SelfSupervise(torch.nn.Module):
    def __init__(self, in_features, out_features, subsample=100):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.subsample = subsample

    def forward(self, y_hat, y):
        idxs = torch.randint(high=y_hat.shape[-3], size=(self.subsample, ), device=y.device)
        y = y[..., idxs, 1:, :].contiguous()
        y_hat = y_hat[..., idxs, :-1, :].contiguous()
        y_hat = self.fc(y_hat)
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=y.detach().mean().pow(-1)
        )(y_hat, y)
        accuracy = ((y_hat.sigmoid() > 0.5) == y).float().mean()
        return loss 