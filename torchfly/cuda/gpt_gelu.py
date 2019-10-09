import math
import os.path
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "csrc/gpt_gelu_cuda.cu"
)

# JIT compiler
_gelu = load(
    name="gelu",
    sources=[_path],
    extra_cflags=['-O3'],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class _GPT_GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        Y = _gelu.gpt_gelu_forward(X)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X = ctx.saved_tensors[0]
        dX = _gelu.gpt_gelu_backward(dY, X)
        return dX

gpt_gelu = _GPT_GELU.apply