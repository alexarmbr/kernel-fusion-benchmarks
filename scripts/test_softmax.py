import torch
from myops import softmax_cuda

BATCH_SIZE = 2
VECTOR_SIZE = 10
output_filename = f"softmax_benchmark_batchsize{BATCH_SIZE}.csv"

x = torch.randn((BATCH_SIZE, VECTOR_SIZE), dtype=torch.float32, device="cuda:0")
y_gt = torch.nn.functional.softmax(x.clone(), dim=1)
y = softmax_cuda(x.clone())
import pdb; pdb.set_trace()
assert torch.allclose(y, y_gt, atol=1e-6)