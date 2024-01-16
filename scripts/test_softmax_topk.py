import torch
from torch_ops.softmax_topk import softmax_topk





x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32)
x2 = torch.tensor([[10,9,8,7,6,5,4,3,2,1]], dtype=torch.float32)

print(softmax_topk(x, 3))
print(softmax_topk(x2, 3))