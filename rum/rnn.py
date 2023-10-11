from functools import partial
import torch

class GRU(torch.nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs["batch_first"] = True
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
        batch_shape = input.shape[:-2]
        event_shape_input = input.shape[-2:]
        event_shape_h_0 = h_0.shape[-1:]
        input = input.view(-1, *event_shape_input)
        h_0 = h_0.view(self.num_layers, -1, *event_shape_h_0)
        output, h_n = super().forward(input, h_0)
        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(self.num_layers, *batch_shape, *h_n.shape[-1:])
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
        output, (h, c) = super().forward(input)
        # output = output.view(*batch_shape, *output.shape[-2:])
        # h_n = h_n.view(self.num_layers, *batch_shape, *h_n.shape[-1:])
        output = output.view(*batch_shape, *output.shape[-2:])
        h = h.view(self.num_layers, *batch_shape, *h.shape[-1:])
        c = c.view(self.num_layers, *batch_shape, *c.shape[-1:])
        return output, (h, c)
