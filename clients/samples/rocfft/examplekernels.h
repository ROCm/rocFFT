// Copyright (C) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef EXAMPLEKERNELS_H
#define EXAMPLEKERNELS_H

#include "../../../shared/data_gen_device.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <iostream>

// Kernel for initializing 1D real input data on the GPU.
__global__ void initrdata1(double* x, const size_t Nx, const size_t xstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < Nx)
    {
        const auto pos = idx * xstride;
        x[pos]         = idx + 1;
    }
}

// Kernel for initializing 2D real input data on the GPU.
__global__ void initrdata2(
    double* x, const size_t Nx, const size_t Ny, const size_t xstride, const size_t ystride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny)
    {
        const auto pos = idx * xstride + idy * ystride;
        x[pos]         = idx + idy;
    }
}

// Kernel for initializing 3D real input data on the GPU.
__global__ void initrdata3(double*      x,
                           const size_t Nx,
                           const size_t Ny,
                           const size_t Nz,
                           const size_t xstride,
                           const size_t ystride,
                           const size_t zstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z;
    if(idx < Nx && idy < Ny && idz < Nz)
    {
        const auto pos = idx * xstride + idy * ystride + idz * zstride;
        x[pos]         = cos(cos(idx + 2)) * sin(idy * idy + 1) / (idz + 1);
    }
}

// Kernel for initializing 1D complex data on the GPU.
__global__ void initcdata1(hipDoubleComplex* x, const size_t Nx, const size_t xstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < Nx)
    {
        const auto pos = idx * xstride;
        x[pos].x       = 1 + idx;
        x[pos].y       = 1 + idx;
    }
}

// Kernel for initializing 2D complex input data on the GPU.
__global__ void initcdata2(hipDoubleComplex* x,
                           const size_t      Nx,
                           const size_t      Ny,
                           const size_t      xstride,
                           const size_t      ystride)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny)
    {
        const auto pos = idx * xstride + idy * ystride;
        x[pos].x       = idx + 1;
        x[pos].y       = idy + 1;
    }
}

// Kernel for initializing 3D complex input data on the GPU.
__global__ void initcdata3(hipDoubleComplex* x,
                           const size_t      Nx,
                           const size_t      Ny,
                           const size_t      Nz,
                           const size_t      xstride,
                           const size_t      ystride,
                           const size_t      zstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z;
    if(idx < Nx && idy < Ny && idz < Nz)
    {
        const auto pos = idx * xstride + idy * ystride + idz * zstride;
        x[pos].x       = idx + 10.0 * idz + 1;
        x[pos].y       = idy + 10;
    }
}

// Helper function for determining grid dimensions
template <typename Tint1, typename Tint2>
Tint1 ceildiv(const Tint1 nominator, const Tint2 denominator)
{
    return (nominator + denominator - 1) / denominator;
}

// The following functions call the above kernels to initalize the input data for the transform.

void initcomplex_cm(const std::vector<size_t>& length_cm,
                    const std::vector<size_t>& stride_cm,
                    void*                      gpu_in)
{
    switch(length_cm.size())
    {
    case 1:
    {
        const dim3 blockdim(256);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x));
        hipLaunchKernelGGL(initcdata1,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length_cm[0],
                           stride_cm[0]);
        break;
    }
    case 2:
    {
        const dim3 blockdim(64, 64);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x), ceildiv(length_cm[1], blockdim.y));
        hipLaunchKernelGGL(initcdata2,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length_cm[0],
                           length_cm[1],
                           stride_cm[0],
                           stride_cm[1]);
        break;
    }
    case 3:
    {
        const dim3 blockdim(32, 32, 32);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x),
                           ceildiv(length_cm[1], blockdim.y),
                           ceildiv(length_cm[2], blockdim.z));
        hipLaunchKernelGGL(initcdata3,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length_cm[0],
                           length_cm[1],
                           length_cm[2],
                           stride_cm[0],
                           stride_cm[1],
                           stride_cm[2]);
        break;
    }
    default:
        std::cout << "invalid dimension!\n";
        exit(1);
    }
}

// Initialize the real input buffer where the data has lengths given in length and stride given in
// stride.  The device buffer is assumed to have been allocated.
void initreal_cm(const std::vector<size_t>& length_cm,
                 const std::vector<size_t>& stride_cm,
                 void*                      gpu_in)
{
    switch(length_cm.size())
    {
    case 1:
    {
        const dim3 blockdim(256);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x));
        hipLaunchKernelGGL(
            initrdata1, blockdim, griddim, 0, 0, (double*)gpu_in, length_cm[0], stride_cm[0]);
        break;
    }
    case 2:
    {
        const dim3 blockdim(64, 64);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x), ceildiv(length_cm[1], blockdim.y));
        hipLaunchKernelGGL(initrdata2,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (double*)gpu_in,
                           length_cm[0],
                           length_cm[1],
                           stride_cm[0],
                           stride_cm[1]);
        break;
    }
    case 3:
    {
        const dim3 blockdim(32, 32, 32);
        const dim3 griddim(ceildiv(length_cm[0], blockdim.x),
                           ceildiv(length_cm[1], blockdim.y),
                           ceildiv(length_cm[2], blockdim.z));
        hipLaunchKernelGGL(initrdata3,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (double*)gpu_in,
                           length_cm[0],
                           length_cm[1],
                           length_cm[2],
                           stride_cm[0],
                           stride_cm[1],
                           stride_cm[2]);
        break;
    }
    default:
        std::cout << "invalid dimension!\n";
        exit(1);
    }
}

void impose_hermitian_symmetry_cm(const std::vector<size_t>& length,
                                  const std::vector<size_t>& ilength,
                                  const std::vector<size_t>& stride,
                                  void*                      gpu_in)
{
    switch(length.size())
    {
    case 1:
    {
        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_1D_kernel<hipDoubleComplex>,
                           dim3(1),
                           dim3(1),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           stride[0],
                           1,
                           1,
                           length[0] % 2 == 0);
        break;
    }
    case 2:
    {
        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_2D_kernel<hipDoubleComplex>,
                           dim3(256),
                           dim3(ceildiv(ceildiv(ilength[1], 2), 256)),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
                           1,
                           1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);
        break;
    }
    case 3:
    {
        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_3D_kernel<hipDoubleComplex>,
                           dim3(64, 64),
                           dim3(ceildiv(ilength[1], 64), ceildiv(ceildiv(ilength[2], 2), 64)),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
                           1,
                           1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0,
                           length[2] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension");
    }
}

// Initialize the real input buffer where the data has lengths given in length, the transform has
// lengths given in length and stride given in stride.  The device buffer is assumed to have been
// allocated.
void init_hermitiancomplex_cm(const std::vector<size_t>& length,
                              const std::vector<size_t>& ilength,
                              const std::vector<size_t>& stride,
                              void*                      gpu_in)
{
    switch(length.size())
    {
    case 1:
    {
        const dim3 blockdim(256);
        const dim3 griddim(ceildiv(ilength[0], blockdim.x));
        hipLaunchKernelGGL(
            initcdata1, blockdim, griddim, 0, 0, (hipDoubleComplex*)gpu_in, ilength[0], stride[0]);
        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_1D_kernel<hipDoubleComplex>,
                           dim3(1),
                           dim3(1),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           stride[0],
                           1,
                           1,
                           length[0] % 2 == 0);
        break;
    }
    case 2:
    {
        const dim3 blockdim(64, 64);
        const dim3 griddim(ceildiv(ilength[0], blockdim.x), ceildiv(ilength[1], blockdim.y));
        hipLaunchKernelGGL(initcdata2,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           ilength[0],
                           ilength[1],
                           stride[0],
                           stride[1]);
        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_2D_kernel<hipDoubleComplex>,
                           dim3(256),
                           dim3(ceildiv(ceildiv(ilength[1], 2), 256)),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
                           1,
                           1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);
        break;
    }
    case 3:
    {
        const dim3 blockdim(32, 32, 32);
        const dim3 griddim(ceildiv(ilength[0], blockdim.x),
                           ceildiv(ilength[1], blockdim.y),
                           ceildiv(ilength[2], blockdim.z));

        hipLaunchKernelGGL(initcdata3,
                           blockdim,
                           griddim,
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           ilength[0],
                           ilength[1],
                           ilength[2],
                           stride[0],
                           stride[1],
                           stride[2]);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_3D_kernel<hipDoubleComplex>,
                           dim3(64, 64),
                           dim3(ceildiv(ilength[1], 64), ceildiv(ceildiv(ilength[2], 2), 64)),
                           0,
                           0,
                           (hipDoubleComplex*)gpu_in,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
                           1,
                           1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0,
                           length[2] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension");
    }

    impose_hermitian_symmetry_cm(length, ilength, stride, gpu_in);
}

#endif /* EXAMPLEKERNELS_H */
