import dgl

def uniform_random_walk(g, num_samples, length):
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
    nodes = g.nodes()
    nodes = nodes.repeat(num_samples)
    walks, _ = dgl.sampling.random_walk(g=g, nodes=nodes, length=length-1)
    walks = walks.view(num_samples, g.number_of_nodes(), length)
    return walks



