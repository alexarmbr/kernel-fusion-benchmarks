#include <torch/extension.h>

torch::Tensor row_softmax_1(torch::Tensor input) {
  auto output = torch::zeros_like(input);
  auto row_size = input.size(1);
  auto col_size = input.size(0);
  for (int i = 0; i < col_size; i++) {
    auto row = input[i];
    auto max = row.max();
    auto row_exp = (row - max).exp();
    auto row_exp_sum = row_exp.sum();
    output[i] = row_exp / row_exp_sum;
  }
  return output;
}

#include <ATen/ATen.h>


torch::Tensor row_softmax_2(torch::Tensor input)
{
    auto output = torch::zeros_like(input);
    auto col_size = input.size(0);
    at::parallel_for(0, col_size, 0, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; i++) {
            auto row = input[i];
            auto max = row.max();
            auto row_exp = (row - max).exp();
            auto row_exp_sum = row_exp.sum();
            output[i] = row_exp / row_exp_sum;
        }
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_softmax_1", &row_softmax_1, "row softmax (for loop)");
  m.def("row_softmax_2", &row_softmax_2, "row softmax (parallel)");
}