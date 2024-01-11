import torch
from myops import softmax_cuda

from typing import Tuple, Callable
import timeit

BATCH_SIZE = 2000
VECTOR_SIZES = [10**i for i in range(2,7)]
output_filename = f"softmax_benchmark_batchsize{BATCH_SIZE}.csv"
output_data = [("vector_size", "pytorch_time", "online_softmax_time")]
torch.random.manual_seed(1234)

def benchmark_softmax(vector_shape : Tuple, fn : Callable, num_iter = 100, num_warmup = 10):
    """Benchmark softmax function for a given vector shape and function"""
    execution_times = []
    for iter in range(num_warmup + num_iter):
        x = torch.randn(vector_shape, dtype=torch.float32, device="cuda:0")
        if iter < num_warmup:
            fn(x)
        else:
            execution_times.append(timeit.timeit(lambda: fn(x), number=1))
        # if iter==11:
        #     print(f"memory use: {torch.cuda.memory_summary()}")
    return sum(execution_times)/len(execution_times)
        

for vector_size in VECTOR_SIZES:
    print(f"Vector size: {vector_size}")
    pytorch_time = benchmark_softmax((BATCH_SIZE, vector_size), lambda x: torch.nn.functional.softmax(x, dim=1))
    # print(f"Pytorch time: {pytorch_time}")
    online_softmax_time = benchmark_softmax((BATCH_SIZE, vector_size), lambda x: softmax_cuda(x))
    # print(f"softmax_cuda_time: {online_softmax_time}")
    print(f"    speedup: {pytorch_time/online_softmax_time}")
    output_data.append([vector_size, pytorch_time, online_softmax_time])
    
with open(output_filename, "w") as f:
    for row in output_data:
        f.write(",".join([str(x) for x in row]) + "\n")