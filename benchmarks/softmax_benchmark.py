import torch
from myops import row_softmax_1, row_softmax_2

from typing import Tuple, Callable
import timeit

BATCH_SIZE = 100
VECTOR_SIZES = [10**i for i in range(2,7)]
output_filename = f"softmax_benchmark_batchsize{4000}.csv"
output_data = [("vector_size", "pytorch_time", "myops_time", "myops2_time")]
torch.random.manual_seed(1234)

def benchmark_softmax(vector_shape : Tuple, fn : Callable, num_iter = 20, num_warmup = 5):
    """Benchmark softmax function for a given vector shape and function"""
    execution_times = []
    for iter in range(num_warmup + num_iter):
        x = torch.randn(vector_shape) * 1000
        if iter < num_warmup:
            fn(x)
        else:
            execution_times.append(timeit.timeit(lambda: fn(x), number=1))
    return sum(execution_times)/len(execution_times)
        

for vector_size in VECTOR_SIZES:
    print(f"Vector size: {vector_size}")
    pytorch_time = benchmark_softmax((BATCH_SIZE, vector_size), lambda x: torch.nn.functional.softmax(x, dim=1))
    print(f"Pytorch time: {pytorch_time}")
    myops2_time = benchmark_softmax((BATCH_SIZE, vector_size), lambda x: row_softmax_2(x))
    print(f"row softmax 2 time: {myops2_time}")
    print(f"speedup: {pytorch_time/myops2_time}")
    output_data.append([vector_size, pytorch_time, myops2_time])
    
with open(output_filename, "w") as f:
    for row in output_data:
        f.write(",".join([str(x) for x in row]) + "\n")