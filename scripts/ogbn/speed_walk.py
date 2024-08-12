import torch
import dgl

def run():
    from ogb.nodeproppred import DglNodePropPredDataset
    g = DglNodePropPredDataset(name='ogbn-products')[0][0]
    g = g.to('cuda:0')
    idxs = torch.randint(0, g.num_nodes(), (5000,)).cuda()
    from dgl.sampling import random_walk
    import time
    memory = torch.cuda.memory_allocated()
    walks = random_walk(g, idxs, length=4)
    for _ in range(10):
        start = time.time()
        walks = random_walk(g, idxs, length=4)
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_allocated() - memory)
        print(time.time() - start)
    
if __name__ == '__main__':
    run()