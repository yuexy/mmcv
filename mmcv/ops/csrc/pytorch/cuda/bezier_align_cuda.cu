#include "pytorch_cuda_helper.hpp"
#include "bezier_align_cuda_kernel.cuh"

void BezierAlignForwardCUDAKernelLauncher(Tensor input, Tensor beziers, Tensor output,
                                          int pooled_height, int pooled_width,
                                          float spatial_scale) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "bezier_align_forward_cuda_kernel", [&] {
        bezier_align_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                beziers.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                pooled_height, pooled_width,
                static_cast<scalar_t>(spatial_scale),
                channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void BezierAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor beziers,
                                           Tensor grad_input, int pooled_height,
                                           int pooled_width, float spatial_scale) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "bezier_align_backward_cuda_kernel", [&] {
        bezier_align_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                beziers.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
                pooled_height, pooled_width,
                static_cast<scalar_t>(spatial_scale),
                channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
