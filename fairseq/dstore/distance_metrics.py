import torch

def dist_func(dstore, d, k, q, function=None):
    if not function:
        # Default behavior for L2 metric is to recompute distances.
        # Default behavior for IP metric is to return faiss distances.
        
        if dstore.metric_type == 'l2':
            return l2(dstore, k, q)
        return d

    if function == 'dot':
        return dot(dstore, k, q)

    if function == 'do_not_recomp_l2':
        return -1 * d

    raise ValueError("Invalid knn similarity function!")


def l2(dstore, k, q):
    qsize = q.shape
    knns_vecs = torch.from_numpy(dstore.keys[k]).cuda().view(qsize[0], dstore.k, -1)
    if dstore.half:
        knns_vecs = knns_vecs.half()
    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, dstore.k, 1)
    metric = torch.sum((query_vecs - knns_vecs
        #.detach() # REVIEW: why was this detached?
    )**2, dim=2)
    return -1 * metric

def dot(dstore, k, q):
    qsize = q.shape
    metric = (torch.from_numpy(dstore.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)
    return metric