import torch
import dgl
from .random_walk import uniform_random_walk
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
            rnn_kwargs: dict = {},
    ):
        super().__init__()
        self.rnn = rnn(in_features, out_features, **rnn_kwargs)
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
        h = h[walks]
        h0 = torch.zeros(1, *h.shape[:-2], self.out_features)
        y, h = self.rnn(h, h0)
        return y, h
