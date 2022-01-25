#ifndef BEZIER_ALIGN_CUDA_KERNEL_CUH
#define BEZIER_ALIGN_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

template <typename T>
__device__ T bezier_curve(T p0, T p1, T p2, T p3, const T u) {
  return ((1. - u) * (1. - u) * (1. - u) * p0 +
          3. * u * (1. - u) * (1. - u) * p1 +
          3. * u * u * (1. - u) * p2 +
          u * u * u * p3);
}

/*** Forward ***/
template <typename T>
__global__ void bezier_align_forward_cuda_kernel(
    const int nthreads, const T* input, const T* beziers, T* output,
    const int pooled_height, const int pooled_width,
    const T spatial_scale, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size of N * (1 + 8 * 2)
    const T* offset_beziers = beziers + n * 17;
    int bezier_batch_ind = offset_beziers[0];

    // Do not use rounding; this implementation detail is critical
    T points[16];
    for (int i = 0; i < 16; i++) {
      points[i] = offset_beziers[i + 1] * spatial_scale;
    }

    const T* offset_input =
        input + (bezier_batch_ind * channels + c) * height * width;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T y0 = bezier_curve(points[0], points[2],
        points[4], points[6], u);
    const T x0 = bezier_curve(points[1], points[3],
        points[5], points[7], u);
    const T y1 = bezier_curve(points[8], points[10],
        points[12], points[14], u);
    const T x1 = bezier_curve(points[9], points[11],
        points[13], points[15], u);
    const T y = y1 * v + y0 * (1. - v);
    const T x = x1 * v + x0 * (1. - v);

    output[index] =
        bilinear_interpolate(offset_input, height, width, y, x, index);
  }
}

/*** Backward ***/
template <typename T>
__global__ void bezier_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* beziers,
    T* grad_input, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size of N * (1 + 8 * 2)
    const T* offset_beziers = beziers + n * 17;
    int bezier_batch_ind = offset_beziers[0];

    T points[16];
    for (int i = 0; i < 16; i++) {
      points[i] = offset_beziers[i + 1] * spatial_scale;
    }

    T* offset_grad_input =
        grad_input + (bezier_batch_ind * channels + c) * height * width;

    const T grad_output_this_bin = grad_output[index];

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T y0 = bezier_curve(points[0], points[2],
      points[4], points[6], u);
    const T x0 = bezier_curve(points[1], points[3],
        points[5], points[7], u);
    const T y1 = bezier_curve(points[8], points[10],
        points[12], points[14], u);
    const T x1 = bezier_curve(points[9], points[11],
        points[13], points[15], u);
    const T y = y1 * v + y0 * (1. - v);
    const T x = x1 * v + x0 * (1. - v);

    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                  x_low, x_high, y_low, y_high, index);

    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
      atomicAdd(offset_grad_input + y_low * width + x_low,
                static_cast<T>(grad_output_this_bin * w1));
      atomicAdd(offset_grad_input + y_low * width + x_high,
                static_cast<T>(grad_output_this_bin * w2));
      atomicAdd(offset_grad_input + y_high * width + x_low,
                static_cast<T>(grad_output_this_bin * w3));
      atomicAdd(offset_grad_input + y_high * width + x_high,
                static_cast<T>(grad_output_this_bin * w4));
    }
  }
}

#endif  // BEZIER_ALIGN_CUDA_KERNEL_CUH
