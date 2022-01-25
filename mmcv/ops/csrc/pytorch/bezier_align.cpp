#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void bezier_align_forward_impl(Tensor input, Tensor beziers, Tensor output,
                               int pooled_height, int pooled_width,
                               float spatial_scale) {
  DISPATCH_DEVICE_IMPL(bezier_align_forward_impl, input, beziers, output,
                       pooled_height, pooled_width, spatial_scale);
}

void bezier_align_backward_impl(Tensor grad_output, Tensor beziers,
                                Tensor grad_input, int pooled_height,
                                int pooled_width, float spatial_scale) {
  DISPATCH_DEVICE_IMPL(bezier_align_backward_impl, grad_output, beziers,
                       grad_input, pooled_height, pooled_width, spatial_scale);
}

void bezier_align_forward(Tensor input, Tensor beziers, Tensor output,
                          int pooled_height, int pooled_width,
                          float spatial_scale) {
  bezier_align_forward_impl(input, beziers, output, pooled_height,
                            pooled_width, spatial_scale);
}

void bezier_align_backward(Tensor grad_output, Tensor beziers,
                           Tensor grad_input, int pooled_height,
                           int pooled_width, float spatial_scale) {
  bezier_align_backward_impl(grad_output, beziers, grad_input, pooled_height,
                             pooled_width, spatial_scale);
}
