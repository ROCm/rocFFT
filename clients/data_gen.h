// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../shared/gpubuf.h"
#include "../shared/increment.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <iostream>
#include <omp.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#include <vector>

static const unsigned int DATA_GEN_THREADS = 32;

template <typename T>
static inline T DivRoundingUp(T a, T b)
{
    return (a + (b - 1)) / b;
}

// count the number of total iterations for 1-, 2-, and 3-D dimensions
template <typename T1>
size_t count_iters(const T1& i)
{
    return i;
}

template <typename T1>
size_t count_iters(const std::tuple<T1, T1>& i)
{
    return std::get<0>(i) * std::get<1>(i);
}

template <typename T1>
size_t count_iters(const std::tuple<T1, T1, T1>& i)
{
    return std::get<0>(i) * std::get<1>(i) * std::get<2>(i);
}

// Work out how many partitions to break our iteration problem into
template <typename T1>
static size_t compute_partition_count(T1 length)
{
#ifdef BUILD_CLIENTS_TESTS_OPENMP
    // we seem to get contention from too many threads, which slows
    // things down.  particularly noticeable with mix_3D tests
    static const size_t MAX_PARTITIONS = 8;
    size_t              iters          = count_iters(length);
    size_t hw_threads = std::min(MAX_PARTITIONS, static_cast<size_t>(omp_get_num_procs()));
    if(!hw_threads)
        return 1;

    // don't bother threading problem sizes that are too small. pick
    // an arbitrary number of iterations and ensure that each thread
    // has at least that many iterations to process
    static const size_t MIN_ITERS_PER_THREAD = 2048;

    // either use the whole CPU, or use ceil(iters/iters_per_thread)
    return std::min(hw_threads, (iters + MIN_ITERS_PER_THREAD + 1) / MIN_ITERS_PER_THREAD);
#else
    return 1;
#endif
}

// Break a scalar length into some number of pieces, returning
// [(start0, end0), (start1, end1), ...]
template <typename T1>
std::vector<std::pair<T1, T1>> partition_base(const T1& length, size_t num_parts)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");

    // make sure we don't exceed the length
    num_parts = std::min(length, num_parts);

    std::vector<std::pair<T1, T1>> ret(num_parts);
    auto                           partition_size = length / num_parts;
    T1                             cur_partition  = 0;
    for(size_t i = 0; i < num_parts; ++i, cur_partition += partition_size)
    {
        ret[i].first  = cur_partition;
        ret[i].second = cur_partition + partition_size;
    }
    // last partition might not divide evenly, fix it up
    ret.back().second = length;
    return ret;
}

// Returns pairs of startindex, endindex, for 1D, 2D, 3D lengths
template <typename T1>
std::vector<std::pair<T1, T1>> partition_rowmajor(const T1& length)
{
    return partition_base(length, compute_partition_count(length));
}

// Partition on the leftmost part of the tuple, for row-major indexing
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>>
    partition_rowmajor(const std::tuple<T1, T1>& length)
{
    auto partitions = partition_base(std::get<0>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<0>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<0>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
    }
    return ret;
}
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>>
    partition_rowmajor(const std::tuple<T1, T1, T1>& length)
{
    auto partitions = partition_base(std::get<0>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<0>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<2>(ret[i].first)  = 0;
        std::get<0>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
        std::get<2>(ret[i].second) = std::get<2>(length);
    }
    return ret;
}

// Returns pairs of startindex, endindex, for 1D, 2D, 3D lengths
template <typename T1>
std::vector<std::pair<T1, T1>> partition_colmajor(const T1& length)
{
    return partition_base(length, compute_partition_count(length));
}

// Partition on the rightmost part of the tuple, for col-major indexing
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>>
    partition_colmajor(const std::tuple<T1, T1>& length)
{
    auto partitions = partition_base(std::get<1>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<1>(ret[i].first)  = partitions[i].first;
        std::get<0>(ret[i].first)  = 0;
        std::get<1>(ret[i].second) = partitions[i].second;
        std::get<0>(ret[i].second) = std::get<0>(length);
    }
    return ret;
}
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>>
    partition_colmajor(const std::tuple<T1, T1, T1>& length)
{
    auto partitions = partition_base(std::get<2>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<2>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<0>(ret[i].first)  = 0;
        std::get<2>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
        std::get<0>(ret[i].second) = std::get<0>(length);
    }
    return ret;
}

// Specialized computation of index given 1-, 2-, 3- dimension length + stride
template <typename T1, typename T2>
size_t compute_index(T1 length, T2 stride, size_t base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (length * stride) + base;
}

template <typename T1, typename T2>
size_t
    compute_index(const std::tuple<T1, T1>& length, const std::tuple<T2, T2>& stride, size_t base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + base;
}

template <typename T1, typename T2>
size_t compute_index(const std::tuple<T1, T1, T1>& length,
                     const std::tuple<T2, T2, T2>& stride,
                     size_t                        base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + (std::get<2>(length) * std::get<2>(stride)) + base;
}

template <typename Tfloat>
void set_rand_generator_seed(size_t seed, std::complex<Tfloat>& val)
{
    val = std::complex<Tfloat>((Tfloat)seed, 0);
}

template <typename Tfloat>
void set_rand_generator_seed(size_t seed, Tfloat& val)
{
    val = (Tfloat)seed;
}

template <typename Tint1, typename Tdata>
void compute_rand_generator_input(size_t       isize,
                                  size_t       idist,
                                  size_t       nbatch,
                                  const Tint1& whole_length,
                                  const Tint1& istride,
                                  size_t&      num_seeds,
                                  Tdata&       rand_gen_seeds)
{
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    rand_gen_seeds.resize(isize);
    num_seeds = nbatch * partitions.size();

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;

            auto seed = compute_index(index, istride, i_base);

            do
            {
                const auto i = compute_index(index, istride, i_base);
                set_rand_generator_seed(seed, rand_gen_seeds[i]);
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_interleaved_data_kernel(size_t isize, Tfloat gen_max, std::complex<Tfloat>* data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        auto seed = (size_t)data[i].real();

        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        auto                 x = (Tfloat)rocrand(&gen_state) / gen_max;
        auto                 y = (Tfloat)rocrand(&gen_state) / gen_max;
        std::complex<Tfloat> val(x, y);
        data[i] = val;
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS) generate_interleaved_data_kernel(
    size_t isize, size_t seed, Tfloat gen_max, std::complex<Tfloat>* data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        auto                 x = (Tfloat)rocrand(&gen_state) / gen_max;
        auto                 y = (Tfloat)rocrand(&gen_state) / gen_max;
        std::complex<Tfloat> val(x, y);
        data[i] = val;
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_planar_data_kernel(size_t isize, Tfloat gen_max, Tfloat* real_data, Tfloat* imag_data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        auto seed = (size_t)real_data[i];

        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        real_data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
        imag_data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS) generate_planar_data_kernel(
    size_t isize, size_t seed, Tfloat gen_max, Tfloat* real_data, Tfloat* imag_data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        real_data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
        imag_data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_real_data_kernel(size_t isize, Tfloat gen_max, Tfloat* data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        auto seed = (size_t)data[i];

        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
    }
}

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_real_data_kernel(size_t isize, size_t seed, Tfloat gen_max, Tfloat* data)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < isize)
    {
        rocrand_state_xorwow gen_state;
        rocrand_init(seed, i, 0, &gen_state);

        data[i] = (Tfloat)rocrand(&gen_state) / gen_max;
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

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_1(std::complex<Tfloat>* x,
                                            const size_t          Nx,
                                            const size_t          xstride,
                                            const size_t          dist,
                                            const size_t          nbatch,
                                            const bool            Nxeven)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < nbatch)
    {
        idx *= dist;

        // The DC mode must be real-valued.
        x[idx].imag(0);

        if(Nxeven)
        {
            // Nyquist mode
            auto pos = idx + (Nx / 2) * xstride;
            x[pos].imag(0);
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
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

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

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_2(std::complex<Tfloat>* x,
                                            const size_t          Nx,
                                            const size_t          Ny,
                                            const size_t          xstride,
                                            const size_t          ystride,
                                            const size_t          dist,
                                            const size_t          nbatch,
                                            const bool            Nxeven,
                                            const bool            Nyeven)
{
    auto       idx = blockIdx.y * blockDim.y + threadIdx.y;
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;

    if(idy < (Ny / 2 + 1) && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride;
        auto cpos = idx + ((Ny - idy) % Ny) * ystride;

        auto val = x[pos];

        // DC mode:
        if(idy == 0)
            val.imag(0);

        // Axes need to be symmetrized:
        if(idy > 0 && idy < (Ny + 1) / 2)
            val = std::conj(val);

        // y-Nyquist
        if(Nyeven && idy == Ny / 2)
            val.imag(0);

        x[cpos] = val;

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;

            val = x[pos];

            // DC mode:
            if(idy == 0)
                val.imag(0);

            // Axes need to be symmetrized:
            if(idy > 0 && idy < (Ny + 1) / 2)
                val = std::conj(val);

            // y-Nyquist
            if(Nyeven && idy == Ny / 2)
                val.imag(0);

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
    auto       idx = blockIdx.y * blockDim.y + threadIdx.y;
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;

    if(idy < (Ny / 2 + 1) && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride;
        auto cpos = idx + ((Ny - idy) % Ny) * ystride;

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

template <typename Tfloat>
__global__ static void __launch_bounds__(DATA_GEN_THREADS* DATA_GEN_THREADS* DATA_GEN_THREADS)
    impose_hermitian_symmetry_interleaved_3(std::complex<Tfloat>* x,
                                            const size_t          Nx,
                                            const size_t          Ny,
                                            const size_t          Nz,
                                            const size_t          xstride,
                                            const size_t          ystride,
                                            const size_t          zstride,
                                            const size_t          dist,
                                            const size_t          nbatch,
                                            const bool            Nxeven,
                                            const bool            Nyeven,
                                            const bool            Nzeven)
{
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idz = blockIdx.y * blockDim.y + threadIdx.y;
    auto       idx = blockIdx.z * blockDim.z + threadIdx.z;

    if(idy < Ny && idz < Nz && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride + idz * zstride;
        auto cpos = idx + ((Ny - idy) % Ny) * ystride + ((Nz - idz) % Nz) * zstride;

        // Origin
        if(idy == 0 && idz == 0)
        {
            x[pos].imag(0);
        }

        // y-Nyquist
        if(Nyeven && idy == Ny / 2 && idz == 0)
        {
            x[pos].imag(0);
        }

        // z-Nyquist
        if(Nzeven && idz == Nz / 2 && idy == 0)
        {
            x[pos].imag(0);
        }

        // yz-Nyquist
        if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
        {
            x[pos].imag(0);
        }

        // z-axis
        if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
            x[cpos] = std::conj(x[pos]);

        // y-Nyquist axis
        if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
            x[cpos] = std::conj(x[pos]);

        // y-axis
        if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
            x[cpos] = std::conj(x[pos]);

        // z-Nyquist axis
        if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
            x[cpos] = std::conj(x[pos]);

        // yz plane
        if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
            x[cpos] = std::conj(x[pos]);

        if(Nxeven)
        {
            pos += (Nx / 2) * xstride;
            cpos += (Nx / 2) * xstride;
            // Origin
            if(idy == 0 && idz == 0)
                x[pos].imag(0);

            // y-Nyquist
            if(Nyeven && idy == Ny / 2 && idz == 0)
                x[pos].imag(0);

            // z-Nyquist
            if(Nzeven && idz == Nz / 2 && idy == 0)
                x[pos].imag(0);

            // yz-Nyquist
            if(Nyeven && Nzeven && idy == Ny / 2 && idz == Nz / 2)
                x[pos].imag(0);

            // z-axis
            if(idy == 0 && idz > 0 && idz < (Nz + 1) / 2)
                x[cpos] = std::conj(x[pos]);

            // y-Nyquist axis
            if(Nyeven && idy == Ny / 2 && idz > 0 && idz < (Nz + 1) / 2)
                x[cpos] = std::conj(x[pos]);

            // y-axis
            if(idy > 0 && idy < (Ny + 1) / 2 && idz == 0)
                x[cpos] = std::conj(x[pos]);

            // z-Nyquist axis
            if(Nzeven && idz == Nz / 2 && idy > 0 && idy < (Ny + 1) / 2)
                x[cpos] = std::conj(x[pos]);

            // yz plane
            if(idy > 0 && idy < (Ny + 1) / 2 && idz > 0 && idz < Nz)
                x[cpos] = std::conj(x[pos]);
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
    const auto idy = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idz = blockIdx.y * blockDim.y + threadIdx.y;
    auto       idx = blockIdx.z * blockDim.z + threadIdx.z;

    if(idy < Ny && idz < Nz && idx < nbatch)
    {
        idx *= dist;

        auto pos  = idx + idy * ystride + idz * zstride;
        auto cpos = idx + ((Ny - idy) % Ny) * ystride + ((Nz - idz) % Nz) * zstride;

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

template <typename Tdata>
inline void init_kernel_args(size_t                    isize,
                             size_t                    num_seeds,
                             const std::vector<Tdata>& rand_gen_seeds,
                             Tdata*                    input_data)
{
    if(num_seeds > 1)
    {
        auto input_data_bytes = isize * sizeof(Tdata);

        if(hipMemcpy(
               input_data, (Tdata*)rand_gen_seeds.data(), input_data_bytes, hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("Failure in hipMemcpy");
    }
}

template <typename Tfloat>
inline void generate_interleaved_data(const std::vector<std::complex<Tfloat>>& rand_gen_seeds,
                                      size_t                                   num_seeds,
                                      std::complex<Tfloat>*                    input_data)
{
    auto isize = rand_gen_seeds.size();
    init_kernel_args<std::complex<Tfloat>>(isize, num_seeds, rand_gen_seeds, input_data);

    auto blockSize       = DATA_GEN_THREADS;
    auto numBlocks_setup = DivRoundingUp<size_t>(isize, blockSize);
    auto gen_max         = (Tfloat)std::numeric_limits<unsigned int>::max();

    if(num_seeds == 1)
    {
        hipLaunchKernelGGL(generate_interleaved_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           (size_t)rand_gen_seeds.at(0).real(),
                           gen_max,
                           input_data);
    }
    else
    {
        hipLaunchKernelGGL(generate_interleaved_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           gen_max,
                           input_data);
    }
}

template <typename Tfloat>
inline void generate_planar_data(const std::vector<Tfloat>& rand_gen_seeds,
                                 size_t                     num_seeds,
                                 Tfloat*                    real_data,
                                 Tfloat*                    imag_data)
{
    auto isize = rand_gen_seeds.size();
    init_kernel_args<Tfloat>(isize, num_seeds, rand_gen_seeds, real_data);

    auto blockSize       = DATA_GEN_THREADS;
    auto numBlocks_setup = DivRoundingUp<size_t>(isize, blockSize);
    auto gen_max         = (Tfloat)std::numeric_limits<unsigned int>::max();

    if(num_seeds == 1)
    {
        hipLaunchKernelGGL(generate_planar_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           (size_t)rand_gen_seeds.at(0),
                           gen_max,
                           real_data,
                           imag_data);
    }
    else
    {
        hipLaunchKernelGGL(generate_planar_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           gen_max,
                           real_data,
                           imag_data);
    }
}

template <typename Tfloat>
inline void generate_real_data(const std::vector<Tfloat>& rand_gen_seeds,
                               size_t                     num_seeds,
                               Tfloat*                    input_data)
{
    auto isize = rand_gen_seeds.size();
    init_kernel_args<Tfloat>(isize, num_seeds, rand_gen_seeds, input_data);

    auto blockSize       = DATA_GEN_THREADS;
    auto numBlocks_setup = DivRoundingUp<size_t>(isize, blockSize);
    auto gen_max         = (Tfloat)std::numeric_limits<unsigned int>::max();

    if(num_seeds == 1)
    {
        hipLaunchKernelGGL(generate_real_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           (size_t)rand_gen_seeds.at(0),
                           gen_max,
                           input_data);
    }
    else
    {
        hipLaunchKernelGGL(generate_real_data_kernel<Tfloat>,
                           dim3(numBlocks_setup),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           0, // stream
                           isize,
                           gen_max,
                           input_data);
    }
}

template <typename Tfloat>
void impose_hermitian_symmetry_interleaved(const std::vector<size_t>& length,
                                           const std::vector<size_t>& ilength,
                                           const std::vector<size_t>& stride,
                                           size_t                     dist,
                                           size_t                     batch,
                                           size_t                     input_sz,
                                           std::complex<Tfloat>*      input_data)
{
    auto blockSize = DATA_GEN_THREADS;

    switch(length.size())
    {
    case 1:
    {
        const auto gridDim  = dim3(blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(batch, blockSize));

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_1<Tfloat>,
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

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_2<Tfloat>,
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

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_3<Tfloat>,
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
}

template <typename Tfloat>
void impose_hermitian_symmetry_planar(const std::vector<size_t>& length,
                                      const std::vector<size_t>& ilength,
                                      const std::vector<size_t>& stride,
                                      size_t                     dist,
                                      size_t                     batch,
                                      Tfloat*                    input_data_real,
                                      Tfloat*                    input_data_imag)
{
    auto blockSize = DATA_GEN_THREADS;

    switch(length.size())
    {
    case 1:
    {
        const auto gridDim  = dim3(blockSize);
        const auto blockDim = dim3(DivRoundingUp<size_t>(batch, blockSize));

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
}

#endif // DATA_GEN_H