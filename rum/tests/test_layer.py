import torch
import dgl

def test_layer_forward():
    from rum.layers import RUMLayer
    g = dgl.graph(([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]))
    layer = RUMLayer(in_features=16, out_features=8, num_samples=2, length=3)
    h = torch.ones(6, 16)
    h = layer(g, h)
    assert h.shape == (6, 8)