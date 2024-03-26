import dgl
import torch
from functools import partial

def uniform_random_walk(g, num_samples, length, subsample=None):
    """
    Random walk on a graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk.

    Returns
    -------
    walks : Tensor
        The random walks.
    """
    if subsample is None:
        nodes = g.nodes()
        num_nodes = g.number_of_nodes()
        nodes = nodes.repeat(num_samples)
    else:
        nodes = subsample.repeat(num_samples)
        num_nodes = subsample.size(0)
    walks, eids, _ = dgl.sampling.random_walk(g=g, nodes=nodes, length=length-1, return_eids=True)
    walks = walks.view(num_samples, num_nodes, length)
    eids = eids.view(num_samples, num_nodes, length-1)
    return walks, eids

# @torch.jit.trace(example_inputs=(torch.zeros(10, 10, 10)))

def uniqueness(walk):
    """
    Compute the uniqueness of a random walk.

    Parameters
    ----------
    walk : Tensor
        The random walk.

    Returns
    -------
    uniqueness : Tensor
        The uniqueness of the random walk.
    """
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal

