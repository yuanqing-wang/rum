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
            dropout: float = 0.5,
            rnn: torch.nn.Module = GRU,
            random_walk: callable = uniform_random_walk,
            **kwargs
    ):
        super().__init__()
        # out_features = out_features // 2
        self.rnn = rnn(in_features + out_features, out_features, **kwargs)
        self.rnn_walk = rnn(length, out_features, **kwargs)
        self.fc = torch.nn.Linear(length, length, bias=False)
        self.fc_self = torch.nn.Linear(in_features, out_features, bias=False)
        self.in_features = in_features
        self.out_features = out_features
        self.random_walk = random_walk
        self.num_samples = num_samples
        self.length = length
        self.dropout = torch.nn.Dropout(dropout)
        self.self_supervise = SelfSupervise(in_features, original_features, self.rnn)

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
        uniqueness_walk = torch.nn.functional.one_hot(
            uniqueness_walk, num_classes=self.length
        ).float()
        h = h[walks]
        h = self.dropout(h)
        h0 = torch.zeros(self.rnn_walk.num_layers, *h.shape[:-2], self.out_features, device=h.device)
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0)
        _, loss = self.self_supervise(h, y0[walks], h_walk, y_walk)
        h = torch.cat([h, y_walk], dim=-1)
        y, h = self.rnn(h, h_walk)
        # y = y.mean(-2)
        h = h.mean(0)
        # h = torch.cat([y, h], dim=-1)
        # h = self.dropout(h)
        return h, loss
    
    def train_self_supervised(self, g, h, y0):
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
        h = self.dropout(h)
        h0 = torch.zeros(self.rnn_walk.num_layers, *h.shape[:-2], self.out_features, device=h.device)
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0)
        h, loss = self.self_supervise(h, y0[walks], h_walk, y_walk)
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
    def __init__(self, in_features, out_features, rnn):
        super().__init__()
        self.rnn = rnn
        self.fc = torch.nn.Linear(in_features, out_features)
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, h, y, h_walk, y_walk):
        h = torch.cat([h, y_walk], dim=-1)
        h = h[:, :, :-1, ...].contiguous()
        y = y[:, :, 1:, ...].contiguous()
        y_hat, h = self.rnn(h, h_walk)
        y_hat = self.fc(y_hat)
        loss = self.loss_fn(y_hat, y)
        return h, loss 


