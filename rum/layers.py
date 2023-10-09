import torch
import dgl
from .random_walk import uniform_random_walk, uniqueness
from .rnn import GRU

class RUMLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_samples: int,
            length: int,
            rnn: torch.nn.Module = GRU,
            random_walk: callable = uniform_random_walk,
            **kwargs
    ):
        super().__init__()
        out_features = out_features // 2
        self.rnn = rnn(in_features + length, out_features, **kwargs)
        self.att = torch.nn.MultiheadAttention(
            embed_dim=out_features * 2, 
            num_heads=4, 
            dropout=0.5,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.random_walk = random_walk
        self.num_samples = num_samples
        self.length = length


    def forward(self, g, h):
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
        uniqueness_walk = torch.nn.functional.one_hot(
            uniqueness_walk, num_classes=self.length
        )
        h = h[walks]
        h = torch.cat([h, uniqueness_walk], dim=-1)
        h0 = torch.zeros(self.rnn.num_layers, *h.shape[:-2], self.out_features, device=h.device)
        y, h = self.rnn(h, h0)
        y = y[..., -1, :]# .mean(0)
        h = h.mean(0)
        h = torch.cat([y, h], dim=-1)
        h = self.att(h, h, h, need_weights=False)[0]
        h = h.mean(0)
        return h
