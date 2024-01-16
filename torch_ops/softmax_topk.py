import torch

def softmax_topk(x, K):
    probabilities = torch.nn.functional.softmax(x, dim=1)
    topk = torch.topk(probabilities, k=K, dim=1)
    return topk