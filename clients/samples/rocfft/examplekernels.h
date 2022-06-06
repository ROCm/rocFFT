// Copyright (C) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef __EXAMPLEKERNELS_H__
#define __EXAMPLEKERNELS_H__

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
__global__ void initcdata1(double2* x, const size_t Nx, const size_t xstride)
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
__global__ void initcdata2(
    double2* x, const size_t Nx, const size_t Ny, const size_t xstride, const size_t ystride)
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
__global__ void initcdata3(double2*     x,
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
        x[pos].x       = idx + 10.0 * idz + 1;
        x[pos].y       = idy + 10;
    }
}

// For complex-to-real transforms, the input data must be Hermitiam-symmetric.
// That is, u_k is the complex conjugate of u_{-k}, where k is the wavevector in Fourier
// space.  For multi-dimensional data, this means that we only need to store a bit more
// than half of the complex values; the rest are redundant.  However, there are still
// some restrictions:
// * the origin and Nyquist value(s) must be real-valued
// * some of the remaining values are still redundant, and you might get different results
//   than you expect if the values don't agree.
// Below are some example kernels which impose Hermitian symmetry on a complex array
// of the given dimensions.

// Kernel for imposing Hermitian symmetry on 1D complex data on the GPU.
__global__ void
    impose_hermitian_symmetry1(double2* x, const size_t Nx, const size_t xstride, const bool Nxeven)
{
    // The DC mode must be real-valued.
    x[0].y = 0.0;
    ;

    if(Nxeven)
    {
        // Nyquist mode
        auto pos = (Nx / 2) * xstride;
        x[pos].y = 0.0;
    }
}

// Kernel for imposing Hermitian symmetry on 2D complex data on the GPU.
__global__ void impose_hermitian_symmetry2(double2*     x,
                                           const size_t Nx,
                                           const size_t Ny,
                                           const size_t xstride,
                                           const size_t ystride,
                                           const bool   Nxeven,
                                           const bool   Nyeven)
{
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;
    if(idy < Ny / 2 + 1)
    {
        auto pos  = idy * ystride;
        auto cpos = ((Ny - idy) % Ny) * ystride;

        auto val = x[pos];

        // DC mode:
        if(idy == 0)
        {
            val.y = 0.0;
        }

        // Axes need to be symmetrized:
        if(idy > 0 && idy < (Ny + 1) / 2)
        {
            val.y = -val.y;
        }

        // y-Nyquist
        if(Nyeven && idy == Ny / 2)
        {
            val.y = 0.0;
        }

        x[cpos] = val;

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;

            val = x[pos];

            // DC mode:
            if(idy == 0)
            {
                val.y = 0.0;
            }

            // Axes need to be symmetrized:
            if(idy > 0 && idy < (Ny + 1) / 2)
            {
                val.y = -val.y;
            }

            // y-Nyquist
            if(Nyeven && idy == Ny / 2)
            {
                val.y = 0.0;
            }

            x[cpos] = val;
        }
    }
}

// Kernel for imposing Hermitian symmetry on 3D complex data on the GPU.
__global__ void impose_hermitian_symmetry3(double2*     x,
                                           const size_t Nx,
                                           const size_t Ny,
                                           const size_t Nz,
                                           const size_t xstride,
                                           const size_t ystride,
                                           const size_t zstride,
                                           const bool   Nxeven,
                                           const bool   Nyeven,
                                           const bool   Nzeven)
{
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idz = blockIdx.y * blockDim.y + threadIdx.y;

    if(idy < Ny && idz < Nz)
    {
        auto pos  = idy * ystride + idz * zstride;
        auto cpos = ((Ny - idy) % Ny) * ystride + ((Nz - idz) % Nz) * zstride;

        // Origin
        if(idy == 0 && idz == 0)
        {
            x[pos].y = 0;
        }

        // y-Nyquist
        if(Nyeven && idy == Ny / 2 && idz == 0)
        {
            x[pos].y = 0;
        }

        // z-Nyquist
        if(Nzeven && idz == Nz / 2 && idy == 0)
        {
            x[pos].y = 0;
        }

        // yz-Nyquist
        if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
        {
            x[pos].y = 0;
        }

        // z-axis
        if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
        {
            x[cpos].x = x[pos].x;
            x[cpos].y = -x[pos].y;
        }
        // y-Nyquist axis
        if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
        {
            x[cpos].x = x[pos].x;
            x[cpos].y = -x[pos].y;
        }

        // y-axis
        if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
        {
            x[cpos].x = x[pos].x;
            x[cpos].y = -x[pos].y;
        }
        // z-Nyquist axis
        if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
        {
            x[cpos].x = x[pos].x;
            x[cpos].y = -x[pos].y;
        }

        // yz plane
        if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
        {
            x[cpos].x = x[pos].x;
            x[cpos].y = -x[pos].y;
        }
        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;
            // Origin
            if(idy == 0 && idz == 0)
            {
                x[pos].y = 0;
            }

            // y-Nyquist
            if(Nyeven && idy == Ny / 2 && idz == 0)
            {
                x[pos].y = 0;
            }

            // z-Nyquist
            if(Nzeven && idz == Nz / 2 && idy == 0)
            {
                x[pos].y = 0;
            }

            // yz-Nyquist
            if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
            {
                x[pos].y = 0;
            }

            // z-axis
            if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
            {
                x[cpos].x = x[pos].x;
                x[cpos].y = -x[pos].y;
            }
            // y-Nyquist axis
            if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
            {
                x[cpos].x = x[pos].x;
                x[cpos].y = -x[pos].y;
            }

            // y-axis
            if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
            {
                x[cpos].x = x[pos].x;
                x[cpos].y = -x[pos].y;
            }
            // z-Nyquist axis
            if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
            {
                x[cpos].x = x[pos].x;
                x[cpos].y = -x[pos].y;
            }

            // yz plane
            if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
            {
                x[cpos].x = x[pos].x;
                x[cpos].y = -x[pos].y;
            }
        }
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
        hipLaunchKernelGGL(
            initcdata1, blockdim, griddim, 0, 0, (double2*)gpu_in, length_cm[0], stride_cm[0]);
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
                           (double2*)gpu_in,
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
                           (double2*)gpu_in,
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
        hipLaunchKernelGGL(impose_hermitian_symmetry1,
                           dim3(1),
                           dim3(1),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           stride[0],
                           length[0] % 2 == 0);
        break;
    }
    case 2:
    {
        hipLaunchKernelGGL(impose_hermitian_symmetry2,
                           dim3(256),
                           dim3(ceildiv(ceildiv(ilength[1], 2), 256)),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);
        break;
    }
    case 3:
    {
        hipLaunchKernelGGL(impose_hermitian_symmetry3,
                           dim3(64, 64),
                           dim3(ceildiv(ilength[1], 64), ceildiv(ceildiv(ilength[2], 2), 64)),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
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
            initcdata1, blockdim, griddim, 0, 0, (double2*)gpu_in, ilength[0], stride[0]);
        hipLaunchKernelGGL(impose_hermitian_symmetry1,
                           dim3(1),
                           dim3(1),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           stride[0],
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
                           (double2*)gpu_in,
                           ilength[0],
                           ilength[1],
                           stride[0],
                           stride[1]);
        hipLaunchKernelGGL(impose_hermitian_symmetry2,
                           dim3(256),
                           dim3(ceildiv(ceildiv(ilength[1], 2), 256)),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
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
                           (double2*)gpu_in,
                           ilength[0],
                           ilength[1],
                           ilength[2],
                           stride[0],
                           stride[1],
                           stride[2]);

        hipLaunchKernelGGL(impose_hermitian_symmetry3,
                           dim3(64, 64),
                           dim3(ceildiv(ilength[1], 64), ceildiv(ceildiv(ilength[2], 2), 64)),
                           0,
                           0,
                           (double2*)gpu_in,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
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

#endif
