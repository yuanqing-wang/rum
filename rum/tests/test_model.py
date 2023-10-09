import torch
import dgl

def test_model_forward():
    from rum.models import RUMModel
    g = dgl.graph(([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]))
    model = RUMModel(in_features=16, out_features=8, hidden_features=12, depth=2, num_samples=2, length=3)
    h = torch.ones(6, 16)
    h = model(g, h)
    assert h.shape == (6, 8)