import torch
import dgl
from rum.random_walk import uniform_random_walk

def test_shape():
    g = dgl.graph(([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]))
    walks = uniform_random_walk(g, 2, 3)
    assert walks.shape == (2, 6, 3)