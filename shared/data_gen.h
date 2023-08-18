// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DATA_GEN_H
#define DATA_GEN_H

// rocRAND can generate warnings if inline asm is not available for
// some architectures.  data generation isn't performance-critical,
// so just disable inline asm to prevent the warnings.
#define ROCRAND_DISABLE_INLINE_ASM

#include "../shared/arithmetic.h"
#include "../shared/device_properties.h"
#include "../shared/gpubuf.h"
#include "../shared/rocfft_complex.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#include <limits>
#include <vector>

static const unsigned int DATA_GEN_THREADS    = 32;
static const unsigned int DATA_GEN_GRID_Y_MAX = 64;

template <typename T>
struct input_val_1D
{
    T val1;
};

template <typename T>
struct input_val_2D
{
    T val1;
    T val2;
};

template <typename T>
struct input_val_3D
{
    T val1;
    T val2;
    T val3;
};

template <typename T>
static input_val_1D<T> get_input_val(const T& val)
{
    return input_val_1D<T>{val};
}

template <typename T>
static input_val_2D<T> get_input_val(const std::tuple<T, T>& val)
{
    return input_val_2D<T>{std::get<0>(val), std::get<1>(val)};
}

template <typename T>
static input_val_3D<T> get_input_val(const std::tuple<T, T, T>& val)
{
    return input_val_3D<T>{std::get<0>(val), std::get<1>(val), std::get<2>(val)};
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_1D<T>& length, const input_val_1D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + base;
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_2D<T>& length, const input_val_2D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + (length.val2 * stride.val2) + base;
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_3D<T>& length, const input_val_3D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + (length.val2 * stride.val2) + (length.val3 * stride.val3)
           + base;
}

template <typename T>
static inline input_val_1D<T> make_zero_length(const input_val_1D<T>& whole_length)
{
    return input_val_1D<T>{0};
}

template <typename T>
static inline input_val_2D<T> make_zero_length(const input_val_2D<T>& whole_length)
{
    return input_val_2D<T>{0, 0};
}

template <typename T>
static inline input_val_3D<T> make_zero_length(const input_val_3D<T>& whole_length)
{
    return input_val_3D<T>{0, 0, 0};
}

template <typename T>
__device__ static input_val_1D<T> get_length(const size_t i, const input_val_1D<T>& whole_length)
{
    auto xlen = whole_length.val1;

    auto xidx = i % xlen;

    return input_val_1D<T>{xidx};
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_1D<T>& whole_length)
{
    auto xlen = whole_length.val1;

    auto yidx = i / xlen;

    return yidx;
}

template <typename T>
__device__ static input_val_2D<T> get_length(const size_t i, const input_val_2D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;

    auto xidx = i % xlen;
    auto yidx = i / xlen % ylen;

    return input_val_2D<T>{xidx, yidx};
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_2D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;

    auto zidx = i / xlen / ylen;

    return zidx;
}

template <typename T>
__device__ static input_val_3D<T> get_length(const size_t i, const input_val_3D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;
    auto zlen = whole_length.val3;

    auto xidx = i % xlen;
    auto yidx = i / xlen % ylen;
    auto zidx = i / xlen / ylen % zlen;

    return input_val_3D<T>{xidx, yidx, zidx};
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_3D<T>& length)
{
    auto xlen = length.val1;
    auto ylen = length.val2;
    auto zlen = length.val3;

    auto widx = i / xlen / ylen / zlen;

    return widx;
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_interleaved_data_kernel(const Tint             whole_length,
                                     const Tint             zero_length,
                                     size_t                 idist,
                                     size_t                 isize,
                                     const Tint             istride,
                                     rocfft_complex<Treal>* data)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);
        auto i_base   = i_batch * idist;

        auto seed = compute_index(zero_length, istride, i_base);
        auto idx  = compute_index(i_length, istride, i_base);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, idx, 0, &gen_state);

        data[idx].x = hiprand_uniform_double(&gen_state) - 0.5;
        data[idx].y = hiprand_uniform_double(&gen_state) - 0.5;
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_planar_data_kernel(const Tint whole_length,
                                const Tint zero_length,
                                size_t     idist,
                                size_t     isize,
                                const Tint istride,
                                Treal*     real_data,
                                Treal*     imag_data)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);
        auto i_base   = i_batch * idist;

        auto seed = compute_index(zero_length, istride, i_base);
        auto idx  = compute_index(i_length, istride, i_base);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, idx, 0, &gen_state);

        real_data[idx] = hiprand_uniform_double(&gen_state) - 0.5;
        imag_data[idx] = hiprand_uniform_double(&gen_state) - 0.5;
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_real_data_kernel(const Tint whole_length,
                              const Tint zero_length,
                              size_t     idist,
                              size_t     isize,
                              const Tint istride,
                              Treal*     data)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);
        auto i_base   = i_batch * idist;

        auto seed = compute_index(zero_length, istride, i_base);
        auto idx  = compute_index(i_length, istride, i_base);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, idx, 0, &gen_state);

        data[idx] = hiprand_uniform_double(&gen_state) - 0.5;
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

// Kernels for imposing Hermitian symmetry on 1D
// complex (interleaved/planar) data on the GPU.

template <typename Tcomplex>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_1(Tcomplex*    x,
                                            const size_t Nx,
                                            const size_t xstride,
                                            const size_t dist,
                                            const size_t nbatch,
                                            const bool   Nxeven)
{
    auto idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    static_assert(sizeof(idx) == sizeof(size_t));

    if(idx < nbatch)
    {
        idx *= dist;

        // The DC mode must be real-valued.
        x[idx].y = 0.0;

        if(Nxeven)
        {
            // Nyquist mode
            auto pos = idx + (Nx / 2) * xstride;
            x[pos].y = 0.0;
        }
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    impose_hermitian_symmetry_planar_1(Tfloat*      xreal,
                                       Tfloat*      ximag,
                                       const size_t Nx,
                                       const size_t xstride,
                                       const size_t dist,
                                       const size_t nbatch,
                                       const bool   Nxeven)
{
    auto idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    static_assert(sizeof(idx) == sizeof(size_t));

    if(idx < nbatch)
    {
        idx *= dist;

        // The DC mode must be real-valued.
        ximag[idx] = 0;

        if(Nxeven)
        {
            // Nyquist mode
            auto pos   = idx + (Nx / 2) * xstride;
            ximag[pos] = 0;
        }
    }
}

// Kernels for imposing Hermitian symmetry on 2D
// complex (interleaved/planar) data on the GPU.

template <typename Tcomplex>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_2(Tcomplex*    x,
                                            const size_t Nx,
                                            const size_t Ny,
                                            const size_t xstride,
                                            const size_t ystride,
                                            const size_t dist,
                                            const size_t nbatch,
                                            const bool   Nxeven,
                                            const bool   Nyeven)
{
    auto       idx = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const auto idy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    static_assert(sizeof(idx) == sizeof(size_t));
    static_assert(sizeof(idy) == sizeof(size_t));

    if(idy < (Ny / 2 + 1) && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride;
        auto cpos = idx + (idy == 0 ? 0 : (Ny - idy)) * ystride;

        auto val = x[pos];

        // DC mode:
        if(idy == 0)
            val.y = 0.0;

        // Axes need to be symmetrized:
        if(idy > 0 && idy < (Ny + 1) / 2)
            val.y = -val.y;

        // y-Nyquist
        if(Nyeven && idy == Ny / 2)
            val.y = 0.0;

        x[cpos] = val;

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;

            val = x[pos];

            // DC mode:
            if(idy == 0)
                val.y = 0;

            // Axes need to be symmetrized:
            if(idy > 0 && idy < (Ny + 1) / 2)
                val.y = -val.y;

            // y-Nyquist
            if(Nyeven && idy == Ny / 2)
                val.y = 0;

            x[cpos] = val;
        }
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_planar_2(Tfloat*      xreal,
                                       Tfloat*      ximag,
                                       const size_t Nx,
                                       const size_t Ny,
                                       const size_t xstride,
                                       const size_t ystride,
                                       const size_t dist,
                                       const size_t nbatch,
                                       const bool   Nxeven,
                                       const bool   Nyeven)
{
    auto       idx = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const auto idy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    static_assert(sizeof(idx) == sizeof(size_t));
    static_assert(sizeof(idy) == sizeof(size_t));

    if(idy < (Ny / 2 + 1) && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride;
        auto cpos = idx + (idy == 0 ? 0 : (Ny - idy)) * ystride;

        auto valreal = xreal[pos];
        auto valimag = ximag[pos];

        // DC mode:
        if(idy == 0)
            valimag = 0;

        // Axes need to be symmetrized:
        if(idy > 0 && idy < (Ny + 1) / 2)
            valimag = -valimag;

        // y-Nyquist
        if(Nyeven && idy == Ny / 2)
            valimag = 0;

        xreal[cpos] = valreal;
        ximag[cpos] = valimag;

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;

            valreal = xreal[pos];
            valimag = ximag[pos];

            // DC mode:
            if(idy == 0)
                valimag = 0;

            // Axes need to be symmetrized:
            if(idy > 0 && idy < (Ny + 1) / 2)
                valimag = -valimag;

            // y-Nyquist
            if(Nyeven && idy == Ny / 2)
                valimag = 0;

            xreal[cpos] = valreal;
            ximag[cpos] = valimag;
        }
    }
}

// Kernels for imposing Hermitian symmetry on 3D
// complex (interleaved/planar) data on the GPU.

template <typename Tcomplex>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_3(Tcomplex*    x,
                                            const size_t Nx,
                                            const size_t Ny,
                                            const size_t Nz,
                                            const size_t xstride,
                                            const size_t ystride,
                                            const size_t zstride,
                                            const size_t dist,
                                            const size_t nbatch,
                                            const bool   Nxeven,
                                            const bool   Nyeven,
                                            const bool   Nzeven)
{
    const auto idy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto idz = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto       idx = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    static_assert(sizeof(idx) == sizeof(size_t));
    static_assert(sizeof(idy) == sizeof(size_t));
    static_assert(sizeof(idz) == sizeof(size_t));

    if(idy < Ny && idz < Nz && idx < nbatch)
    {
        idx *= dist;

        auto pos = idx + idy * ystride + idz * zstride;
        auto cpos
            = idx + (idy == 0 ? 0 : (Ny - idy)) * ystride + (idz == 0 ? 0 : (Nz - idz)) * zstride;

        // Origin
        if(idy == 0 && idz == 0)
        {
            x[pos].y = 0.0;
        }

        // y-Nyquist
        if(Nyeven && idy == Ny / 2 && idz == 0)
        {
            x[pos].y = 0.0;
        }

        // z-Nyquist
        if(Nzeven && idz == Nz / 2 && idy == 0)
        {
            x[pos].y = 0.0;
        }

        // yz-Nyquist
        if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
        {
            x[pos].y = 0.0;
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
                x[pos].y = 0.0;

            // y-Nyquist
            if(Nyeven && idy == Ny / 2 && idz == 0)
                x[pos].y = 0.0;

            // z-Nyquist
            if(Nzeven && idz == Nz / 2 && idy == 0)
                x[pos].y = 0.0;

            // yz-Nyquist
            if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
                x[pos].y = 0.0;

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

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_planar_3(Tfloat*      xreal,
                                       Tfloat*      ximag,
                                       const size_t Nx,
                                       const size_t Ny,
                                       const size_t Nz,
                                       const size_t xstride,
                                       const size_t ystride,
                                       const size_t zstride,
                                       const size_t dist,
                                       const size_t nbatch,
                                       const bool   Nxeven,
                                       const bool   Nyeven,
                                       const bool   Nzeven)
{
    const auto idy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto idz = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto       idx = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    static_assert(sizeof(idx) == sizeof(size_t));
    static_assert(sizeof(idy) == sizeof(size_t));
    static_assert(sizeof(idz) == sizeof(size_t));

    if(idy < Ny && idz < Nz && idx < nbatch)
    {
        idx *= dist;

        auto pos = idx + idy * ystride + idz * zstride;
        auto cpos
            = idx + (idy == 0 ? 0 : (Ny - idy)) * ystride + (idz == 0 ? 0 : (Nz - idz)) * zstride;

        // Origin
        if(idy == 0 && idz == 0)
        {
            ximag[pos] = 0;
        }

        // y-Nyquist
        if(Nyeven && idy == Ny / 2 && idz == 0)
        {
            ximag[pos] = 0;
        }

        // z-Nyquist
        if(Nzeven && idz == Nz / 2 && idy == 0)
        {
            ximag[pos] = 0;
        }

        // yz-Nyquist
        if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
        {
            ximag[pos] = 0;
        }

        // z-axis
        if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
        {
            xreal[cpos] = xreal[pos];
            ximag[cpos] = -ximag[pos];
        }

        // y-Nyquist axis
        if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
        {
            xreal[cpos] = xreal[pos];
            ximag[cpos] = -ximag[pos];
        }

        // y-axis
        if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
        {
            xreal[cpos] = xreal[pos];
            ximag[cpos] = -ximag[pos];
        }

        // z-Nyquist axis
        if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
        {
            xreal[cpos] = xreal[pos];
            ximag[cpos] = -ximag[pos];
        }

        // yz plane
        if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
        {
            xreal[cpos] = xreal[pos];
            ximag[cpos] = -ximag[pos];
        }

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;
            // Origin
            if(idy == 0 && idz == 0)
                ximag[pos] = 0;

            // y-Nyquist
            if(Nyeven && idy == Ny / 2 && idz == 0)
                ximag[pos] = 0;

            // z-Nyquist
            if(Nzeven && idz == Nz / 2 && idy == 0)
                ximag[pos] = 0;

            // yz-Nyquist
            if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
                ximag[pos] = 0;

            // z-axis
            if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
            {
                xreal[cpos] = xreal[pos];
                ximag[cpos] = -ximag[pos];
            }

            // y-Nyquist axis
            if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
            {
                xreal[cpos] = xreal[pos];
                ximag[cpos] = -ximag[pos];
            }

            // y-axis
            if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
            {
                xreal[cpos] = xreal[pos];
                ximag[cpos] = -ximag[pos];
            }

            // z-Nyquist axis
            if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
            {
                xreal[cpos] = xreal[pos];
                ximag[cpos] = -ximag[pos];
            }

            // yz plane
            if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
            {
                xreal[cpos] = xreal[pos];
                ximag[cpos] = -ximag[pos];
            }
        }
    }
}

// get grid dimensions for data gen kernel
static dim3 generate_data_gridDim(const size_t isize)
{
    auto blockSize = DATA_GEN_THREADS;
    // total number of blocks needed in the grid
    auto numBlocks_setup = DivRoundingUp<size_t>(isize, blockSize);

    // Total work items per dimension in the grid is counted in
    // uint32_t.  Since each thread initializes one element, very
    // large amounts of data will overflow this total size if we do
    // all this work in one grid dimension, causing launch failure.
    //
    // CUDA also generally allows for effectively unlimited grid X
    // dim, but Y and Z are more limited.
    auto gridDim_y = std::min<unsigned int>(DATA_GEN_GRID_Y_MAX, numBlocks_setup);
    auto gridDim_x = DivRoundingUp<unsigned int>(numBlocks_setup, DATA_GEN_GRID_Y_MAX);
    return {gridDim_x, gridDim_y};
}

template <typename Tint, typename Treal>
inline void generate_interleaved_data(const Tint&            whole_length,
                                      const size_t           idist,
                                      const size_t           isize,
                                      const Tint&            istride,
                                      rocfft_complex<Treal>* input_data,
                                      const hipDeviceProp_t& deviceProp)
{
    auto input_length = get_input_val(whole_length);
    auto zero_length  = make_zero_length(input_length);
    auto input_stride = get_input_val(istride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check(gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generate_interleaved_data_kernel<decltype(input_length), Treal>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        input_length,
        zero_length,
        idist,
        isize,
        input_stride,
        input_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_interleaved_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
inline void generate_planar_data(const Tint&            whole_length,
                                 const size_t           idist,
                                 const size_t           isize,
                                 const Tint&            istride,
                                 Treal*                 real_data,
                                 Treal*                 imag_data,
                                 const hipDeviceProp_t& deviceProp)
{
    auto input_length = get_input_val(whole_length);
    auto zero_length  = make_zero_length(input_length);
    auto input_stride = get_input_val(istride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check(gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(generate_planar_data_kernel<decltype(input_length), Treal>),
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       input_length,
                       zero_length,
                       idist,
                       isize,
                       input_stride,
                       real_data,
                       imag_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_planar_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
inline void generate_real_data(const Tint&            whole_length,
                               const size_t           idist,
                               const size_t           isize,
                               const Tint&            istride,
                               Treal*                 input_data,
                               const hipDeviceProp_t& deviceProp)
{
    auto input_length = get_input_val(whole_length);
    auto zero_length  = make_zero_length(input_length);
    auto input_stride = get_input_val(istride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check(gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(generate_real_data_kernel<decltype(input_length), Treal>),
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       input_length,
                       zero_length,
                       idist,
                       isize,
                       input_stride,
                       input_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_real_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tcomplex>
void impose_hermitian_symmetry_interleaved(const std::vector<size_t>& length,
                                           const std::vector<size_t>& ilength,
                                           const std::vector<size_t>& stride,
                                           size_t                     dist,
                                           size_t                     batch,
                                           Tcomplex*                  input_data,
                                           const hipDeviceProp_t&     deviceProp)
{
    auto blockSize = DATA_GEN_THREADS;

    switch(length.size())
    {
    case 1:
    {
        const auto gridDim  = dim3(blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_1<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[0],
                           stride[0],
                           dist,
                           batch,
                           length[0] % 2 == 0);

        break;
    }
    case 2:
    {
        const auto gridDim  = dim3(blockSize, blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(ilength[0], blockSize),
                                   DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_2<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[1],
                           length[0],
                           stride[1],
                           stride[0],
                           dist,
                           batch,
                           length[1] % 2 == 0,
                           length[0] % 2 == 0);

        break;
    }
    case 3:
    {
        const auto gridDim  = dim3(blockSize, blockSize, blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(ilength[0], blockSize),
                                   DivRoundingUp<size_t>(ilength[1], blockSize),
                                   DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_3<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[2],
                           length[0],
                           length[1],
                           stride[2],
                           stride[0],
                           stride[1],
                           dist,
                           batch,
                           length[2] % 2 == 0,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("impose_hermitian_symmetry_interleaved launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tfloat>
void impose_hermitian_symmetry_planar(const std::vector<size_t>& length,
                                      const std::vector<size_t>& ilength,
                                      const std::vector<size_t>& stride,
                                      size_t                     dist,
                                      size_t                     batch,
                                      Tfloat*                    input_data_real,
                                      Tfloat*                    input_data_imag,
                                      const hipDeviceProp_t&     deviceProp)
{
    auto blockSize = DATA_GEN_THREADS;

    switch(length.size())
    {
    case 1:
    {
        const auto gridDim  = dim3(blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_1<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[0],
                           stride[0],
                           dist,
                           batch,
                           length[0] % 2 == 0);

        break;
    }
    case 2:
    {
        const auto gridDim  = dim3(blockSize, blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(ilength[0], blockSize),
                                   DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_2<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[1],
                           length[0],
                           stride[1],
                           stride[0],
                           dist,
                           batch,
                           length[1] % 2 == 0,
                           length[0] % 2 == 0);

        break;
    }
    case 3:
    {
        const auto gridDim  = dim3(blockSize, blockSize, blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(ilength[0], blockSize),
                                   DivRoundingUp<size_t>(ilength[1], blockSize),
                                   DivRoundingUp<size_t>(batch, blockSize));

        launch_limits_check(gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_3<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[2],
                           length[0],
                           length[1],
                           stride[2],
                           stride[0],
                           stride[1],
                           dist,
                           batch,
                           length[2] % 2 == 0,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("impose_hermitian_symmetry_planar launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

#endif // DATA_GEN_H
