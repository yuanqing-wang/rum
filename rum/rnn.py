from functools import partial
import torch

class GRU(torch.nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs["batch_first"] = True
        # kwargs["bidirectional"] = True
        super().__init__(*args, **kwargs)
    
    def forward(self, input, h_0):
        """Forward pass.

        Parameters
        ----------
        input : Tensor
            The input features.

        h_0 : Tensor
            The initial hidden state.

        Returns
        -------
        output : Tensor
            The output features.

        h_n : Tensor
            The final hidden state.
        """
        num_direction = 2 if self.bidirectional else 1
        batch_shape = input.shape[:-2]
        event_shape_input = input.shape[-2:]
        event_shape_h_0 = h_0.shape[-1:]
        input = input.view(-1, *event_shape_input)
        h_0 = h_0.view(num_direction * self.num_layers, -1, *event_shape_h_0)
        output, h_n = super().forward(input, h_0)
        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(num_direction * self.num_layers, *batch_shape, *h_n.shape[-1:])
        return output, h_n

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs["batch_first"] = True
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass.

        Parameters
        ----------
        input : Tensor
            The input features.

        h_0 : Tensor
            The initial hidden state.

        Returns
        -------
        output : Tensor
            The output features.

        h_n : Tensor
            The final hidden state.
        """
        batch_shape = input.shape[:-2]
        event_shape_input = input.shape[-2:]
        input = input.view(-1, *event_shape_input)
        output, h_n = super().forward(input)
        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(self.num_layers, *batch_shape, *h_n.shape[-1:])
        return output, h_n

class MultiheadAttention(torch.nn.MultiheadAttention):
    def forward(self, h):
        """Forward pass.

        Parameters
        ----------
        h : Tensor
            The input features.

        Returns
        -------
        output : Tensor
            The output features.
        """
        batch_shape = h.shape[:-2]
        event_shape = h.shape[-2:]
        h = h.view(-1, *event_shape)
        output, _ = super().forward(h, h, h)
        output = output.view(*batch_shape, *output.shape[-2:])
        return output