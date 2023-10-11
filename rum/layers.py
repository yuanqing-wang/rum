import torch
import dgl
from .random_walk import uniform_random_walk, uniqueness
from .rnn import GRU, LSTM

class RUMLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_samples: int,
            length: int,
            rnn: torch.nn.Module = LSTM,
            random_walk: callable = uniform_random_walk,
            **kwargs
    ):
        super().__init__()
        # out_features = out_features // 3
        self.rnn = rnn(in_features + length, out_features, **kwargs)
        self.fc = torch.nn.Linear(length, length, bias=False)
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
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = torch.nn.functional.one_hot(
            uniqueness_walk, num_classes=self.length
        ).float()
        h = h[walks]
        uniqueness_walk = self.fc(uniqueness_walk)
        h = torch.cat([h, uniqueness_walk], dim=-1)
        # h0 = torch.zeros(self.rnn.num_layers, *h.shape[:-2], self.out_features, device=h.device)
        # y, h = self.rnn(h, h0)
        y, (h, c) = self.rnn(h)
        print(y.shape)
        return y[..., -1, :]
