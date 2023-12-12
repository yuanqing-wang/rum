import math
import torch
import dgl
from .random_walk import uniform_random_walk, uniqueness
from .rnn import GRU

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
            **kwargs
    ):
        super().__init__()
        # out_features = out_features // 2
        self.rnn = rnn(in_features + 2 * out_features + 1, out_features, **kwargs)
        self.rnn_walk = rnn(2, out_features, bidirectional=True, **kwargs)
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
        num_directions = 2 if self.rnn_walk.bidirectional else 1
        h0 = torch.zeros(self.rnn_walk.num_layers * num_directions, *h.shape[:-2], self.out_features, device=h.device)
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0)
        h_walk = h_walk.mean(0, keepdim=True)
        h = torch.cat([h, y_walk, degrees], dim=-1)
        y, h = self.rnn(h, h_walk)
        if self.training:
            loss = self.self_supervise(y, y0[walks], degrees)
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
        self.fc = torch.nn.Linear(in_features, out_features+1)
        self.out_features = out_features
        self.subsample = subsample

    def forward(self, y_hat, y, degrees):
        idxs = torch.randint(high=y_hat.shape[-3], size=(self.subsample, ), device=y.device)
        y = y[..., idxs, 1:, :].contiguous()
        y_hat = y_hat[..., idxs, :-1, :].contiguous()
        y_hat, degrees_hat = self.fc(y_hat).split([self.out_features, 1], dim=-1)
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=y.detach().mean().pow(-1)
        )(y_hat, y)\
            + 0.001 * torch.nn.L1Loss()(degrees_hat, degrees[..., idxs, :-1, :].contiguous())

        return loss 