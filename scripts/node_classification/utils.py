import torch
from ogb.nodeproppred import DglNodePropPredDataset
from functools import partial

def OGBDataset(name):
    data = DglNodePropPredDataset(name=name)
    g, y = data[0]
    split_idx = data.get_idx_split()
    g.ndata['train_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    g.ndata['val_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    g.ndata['test_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    g.ndata['train_mask'][split_idx['train']] = True
    g.ndata['val_mask'][split_idx['valid']] = True
    g.ndata['test_mask'][split_idx['test']] = True
    g.ndata['label'] = y.squeeze()
    return [g]

Arxiv = partial(OGBDataset, name='ogbn-arxiv')




