#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

#define TORCH_CHECK AT_ASSERTM
// Check if CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
// Check if CUDA Contiguous
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CUDA_NUM_THREADS 512

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define SQRT_2_DIV_PI 0.7978845608028654

#define CUBE(x) x *x *x
#define SQUARE(x) x *x

template <typename T>
__global__ void gpt_gelu_forward_kernel(const int N,
                                        const T *X,
                                        T *Y)
{
    const int index = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
    if (index < N)
    {
        Y[index] = 0.5 * fmaf(X[index],
                              tanhf(SQRT_2_DIV_PI * fmaf(0.044715,
                                                         CUBE(X[index]),
                                                         X[index])),
                              X[index]);
    }
}

template <typename T>
__global__ void gpt_gelu_backward_kernel(const int N,
                                         const T *dY,
                                         const T *X,
                                         T *dX)
{
    const int index = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
    if (index < N)
    {
        const T y = tanhf(SQRT_2_DIV_PI * fmaf(0.044715,
                                               CUBE(X[index]),
                                               X[index]));
        dX[index] = fmaf(fmaf(-X[index], SQUARE(y), X[index]),
                         fmaf(0.10703222440890037, SQUARE(X[index]), 0.044715),
                         1.0 + y) *
                    dY[index] * 0.5;
    }
}

at::Tensor gpt_gelu_forward(const at::Tensor &X)
{
    // Check inputs
    CHECK_CUDA(X);
    CHECK_CONTIGUOUS(X);

    // Create outputs Tensor
    auto Y = at::empty_like(X);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Define grid size
    auto X_size = X.numel();
    const long M = THCCeilDiv(X_size, (long)CUDA_NUM_THREADS);

    // Corner case
    if (X_size == 0)
    {
        THCudaCheck(cudaGetLastError());
        return Y;
    }

    // Dispatch Kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "gpt_gelu_forward", [&] {
        gpt_gelu_forward_kernel<scalar_t><<<M, CUDA_NUM_THREADS, 0, stream>>>(X_size,
                                                                              X.data<scalar_t>(),
                                                                              Y.data<scalar_t>());
    });

    THCudaCheck(cudaGetLastError());
    return Y;
}

at::Tensor gpt_gelu_backward(const at::Tensor &dY,
                             const at::Tensor &X)
{
    // Create outputs Tensor
    auto dX = at::empty_like(X);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Define grid size
    auto X_size = X.numel();
    const long M = THCCeilDiv(X_size, (long)CUDA_NUM_THREADS);

    // Corner case
    if (X_size == 0)
    {
        THCudaCheck(cudaGetLastError());
        return dX;
    }

    // Dispatch Kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "gpt_gelu_backward", [&] {
        gpt_gelu_backward_kernel<scalar_t><<<M, CUDA_NUM_THREADS, 0, stream>>>(X_size,
                                                                               dY.data<scalar_t>(),
                                                                               X.data<scalar_t>(),
                                                                               dX.data<scalar_t>());
    });

    THCudaCheck(cudaGetLastError());
    return dX;
}

// bind everything
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gpt_gelu_forward", &gpt_gelu_forward, "GPT_GELU forward (CUDA)");
    m.def("gpt_gelu_backward", &gpt_gelu_backward, "GPT_GELU backward (CUDA)");
}