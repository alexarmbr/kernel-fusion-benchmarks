#include <cub/cub.cuh>
#include <torch/extension.h>

// from here https://github.com/NVIDIA/online-softmax/blob/master/online_softmax_benchmark.cu#L156
struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
}

torch::Tensor softmax_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.device().is_cuda(), "softmax input must be a CUDA tensor")
    auto output = torch::zeros_like(input);
    const int batch_size = input.size(0);
    const int vector_size = input.size(1);
    const int threadblock_size = 512;

    online_softmax<threadblock_size><<<batch_size, threadblock_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), vector_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_cuda", &softmax_cuda, "softmax (CUDA)");
}