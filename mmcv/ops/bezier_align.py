from tkinter import S
from matplotlib import bezier
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['bezier_align_forward',
                                  'bezier_align_backward'])


class BezierAlignFunction(Function):

    @staticmethod
    def forward(ctx, input, beziers, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()

        assert beziers.size(1) == 17
        output_shape = (beziers.size(0), input.size(1),
                        ctx.output_size[0], ctx.output_size[1])
        output = input.new_zeros(output_shape)
        ext_module.bezier_align_forward(
            input,
            beziers,
            output,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale)

        ctx.save_for_backward(beziers)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        beziers = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        grad_output = grad_output.contiguous()
        ext_module.bezier_align_backward(
            grad_output,
            beziers,
            grad_input,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale)
        return grad_input, None, None, None


bezier_align = BezierAlignFunction.apply


class BezierAlign(nn.Module):
    """
    """

    def __init__(self,
                 output_size,
                 spatial_scale):
        super(BezierAlign, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, input, beziers):
        """
        """
        return bezier_align(input,
                            beziers,
                            self.output_size,
                            self.spatial_scale)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        return s
