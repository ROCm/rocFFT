// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "transpose.h"
#include "kernel_launch.h"
#include "rocfft_hip.h"
#include <iostream>

/// \brief FFT Transpose out-of-place API
/// \details transpose matrix A of size (m row by n cols) to matrix B (n row by m cols)
///    both A and B are in row major
///
/// @param[in]    m size_t.
/// @param[in]    n size_t.
/// @param[in]    A pointer storing batch_count of A matrix on the GPU.
/// @param[inout] B pointer storing batch_count of B matrix on the GPU.
/// @param[in]    count size_t number of matrices processed
template <typename T, typename TA, typename TB, int TRANSPOSE_DIM_X, int TRANSPOSE_DIM_Y>
rocfft_status rocfft_transpose_outofplace_template(size_t      m,
                                                   size_t      n,
                                                   const TA*   A,
                                                   TB*         B,
                                                   void*       twiddles_large,
                                                   size_t      count,
                                                   size_t*     lengths,
                                                   size_t*     stride_in,
                                                   size_t*     stride_out,
                                                   int         twl,
                                                   int         dir,
                                                   int         scheme,
                                                   bool        unit_stride0,
                                                   bool        diagonal,
                                                   size_t      ld_in,
                                                   size_t      ld_out,
                                                   hipStream_t rocfft_stream)
{

    dim3 grid((n - 1) / TRANSPOSE_DIM_X + 1, ((m - 1) / TRANSPOSE_DIM_X + 1), count);
    dim3 threads(TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, 1);

    // working threads match problem sizes, no partial cases
    const bool all = (n % TRANSPOSE_DIM_X == 0) && (m % TRANSPOSE_DIM_X == 0);

    if(scheme == 0)
    {
        // Create a map from the parameters to the templated function
        std::map<std::tuple<int, int, bool, bool, bool>, // TWL, DIR, ALL, UNIT_STRIDE_0, DIAGONAL
                 decltype(&HIP_KERNEL_NAME(transpose_kernel2<T,
                                                             TA,
                                                             TB,
                                                             TRANSPOSE_DIM_X,
                                                             TRANSPOSE_DIM_Y,
                                                             true,
                                                             2,
                                                             -1,
                                                             true,
                                                             true,
                                                             true>))>
            tmap;
        // Fill the map with explicitly instantiated templates:

        // clang-format off
        // twl=0:
        tmap.emplace(
            std::make_tuple(0, -1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, true, true, true>));
        tmap.emplace(
            std::make_tuple(0, -1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, false, true, true>));
        tmap.emplace(
            std::make_tuple(0, -1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, true, false, true>));
        tmap.emplace(
            std::make_tuple(0, -1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, false, false, true>));

        tmap.emplace(
            std::make_tuple(0, 1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, true, true, true>));
        tmap.emplace(
            std::make_tuple(0, 1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, false, true, true>));

        tmap.emplace(
            std::make_tuple(0, 1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, true, false, true>));
        tmap.emplace(
            std::make_tuple(0, 1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, false, false, true>));

        tmap.emplace(
            std::make_tuple(0, -1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, true, true, false>));
        tmap.emplace(
            std::make_tuple(0, -1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, false, true, false>));
        tmap.emplace(
            std::make_tuple(0, -1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, true, false, false>));
        tmap.emplace(
            std::make_tuple(0, -1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, -1, false, false, false>));

        tmap.emplace(
            std::make_tuple(0, 1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, true, true, false>));
        tmap.emplace(
            std::make_tuple(0, 1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, false, true, false>));

        tmap.emplace(
            std::make_tuple(0, 1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, true, false, false>));
        tmap.emplace(
            std::make_tuple(0, 1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 0, 1, false, false, false>));

        // twl=1:
        tmap.emplace(
            std::make_tuple(1, -1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, true, true, true>));
        tmap.emplace(
            std::make_tuple(1, -1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, false, true, true>));
        tmap.emplace(
            std::make_tuple(1, -1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, true, false, true>));
        tmap.emplace(
            std::make_tuple(1, -1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, false, false, true>));

        tmap.emplace(
            std::make_tuple(1, 1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, true, true, true>));
        tmap.emplace(
            std::make_tuple(1, 1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, false, true, true>));

        tmap.emplace(
            std::make_tuple(1, 1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, true, false, true>));
        tmap.emplace(
            std::make_tuple(1, 1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, false, false, true>));

        tmap.emplace(
            std::make_tuple(1, -1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, true, true, false>));
        tmap.emplace(
            std::make_tuple(1, -1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, false, true, false>));
        tmap.emplace(
            std::make_tuple(1, -1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, true, false, false>));
        tmap.emplace(
            std::make_tuple(1, -1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, -1, false, false, false>));

        tmap.emplace(
            std::make_tuple(1, 1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, true, true, false>));
        tmap.emplace(
            std::make_tuple(1, 1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, false, true, false>));

        tmap.emplace(
            std::make_tuple(1, 1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, true, false, false>));
        tmap.emplace(
            std::make_tuple(1, 1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 1, 1, false, false, false>));

        // twl=2:
        tmap.emplace(
            std::make_tuple(2, -1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, true, true, true>));
        tmap.emplace(
            std::make_tuple(2, -1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, false, true, true>));
        tmap.emplace(
            std::make_tuple(2, -1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, true, false, true>));
        tmap.emplace(
            std::make_tuple(2, -1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, false, false, true>));

        tmap.emplace(
            std::make_tuple(2, 1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, true, true, true>));
        tmap.emplace(
            std::make_tuple(2, 1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, false, true, true>));

        tmap.emplace(
            std::make_tuple(2, 1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, true, false, true>));
        tmap.emplace(
            std::make_tuple(2, 1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, false, false, true>));

        tmap.emplace(
            std::make_tuple(2, -1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, true, true, false>));
        tmap.emplace(
            std::make_tuple(2, -1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, false, true, false>));
        tmap.emplace(
            std::make_tuple(2, -1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, true, false, false>));
        tmap.emplace(
            std::make_tuple(2, -1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, -1, false, false, false>));

        tmap.emplace(
            std::make_tuple(2, 1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, true, true, false>));
        tmap.emplace(
            std::make_tuple(2, 1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, false, true, false>));

        tmap.emplace(
            std::make_tuple(2, 1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, true, false, false>));
        tmap.emplace(
            std::make_tuple(2, 1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 2, 1, false, false, false>));

        // twl=3:
        tmap.emplace(
            std::make_tuple(3, -1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, true, true, true>));
        tmap.emplace(
            std::make_tuple(3, -1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, false, true, true>));
        tmap.emplace(
            std::make_tuple(3, -1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, true, false, true>));
        tmap.emplace(
            std::make_tuple(3, -1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, false, false, true>));

        tmap.emplace(
            std::make_tuple(3, 1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, true, true, true>));
        tmap.emplace(
            std::make_tuple(3, 1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, false, true, true>));
        tmap.emplace(
            std::make_tuple(3, 1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, true, false, true>));
        tmap.emplace(
            std::make_tuple(3, 1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, false, false, true>));

        tmap.emplace(
            std::make_tuple(3, -1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, true, true, false>));
        tmap.emplace(
            std::make_tuple(3, -1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, false, true, false>));
        tmap.emplace(
            std::make_tuple(3, -1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, true, false, false>));
        tmap.emplace(
            std::make_tuple(3, -1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, -1, false, false, false>));

        tmap.emplace(
            std::make_tuple(3, 1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, true, true, false>));
        tmap.emplace(
            std::make_tuple(3, 1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, false, true, false>));
        tmap.emplace(
            std::make_tuple(3, 1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, true, false, false>));
        tmap.emplace(
            std::make_tuple(3, 1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 3, 1, false, false, false>));

        // twl=4:
        tmap.emplace(
            std::make_tuple(4, -1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, true, true, true>));
        tmap.emplace(
            std::make_tuple(4, -1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, false, true, true>));
        tmap.emplace(
            std::make_tuple(4, -1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, true, false, true>));
        tmap.emplace(
            std::make_tuple(4, -1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, false, false, true>));

        tmap.emplace(
            std::make_tuple(4, 1, true, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, true, true, true>));
        tmap.emplace(
            std::make_tuple(4, 1, false, true, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, false, true, true>));
        tmap.emplace(
            std::make_tuple(4, 1, true, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, true, false, true>));
        tmap.emplace(
            std::make_tuple(4, 1, false, false, true),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, false, false, true>));


        tmap.emplace(
            std::make_tuple(4, -1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, true, true, false>));
        tmap.emplace(
            std::make_tuple(4, -1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, false, true, false>));
        tmap.emplace(
            std::make_tuple(4, -1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, true, false, false>));
        tmap.emplace(
            std::make_tuple(4, -1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, -1, false, false, false>));

        tmap.emplace(
            std::make_tuple(4, 1, true, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, true, true, false>));
        tmap.emplace(
            std::make_tuple(4, 1, false, true, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, false, true, false>));
        tmap.emplace(
            std::make_tuple(4, 1, true, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, true, false, false>));
        tmap.emplace(
            std::make_tuple(4, 1, false, false, false),
            &HIP_KERNEL_NAME(
                transpose_kernel2<T, TA, TB, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true, 4, 1, false, false, false>));
        // clang-format on

        // Tuple containing template parameters for transpose TWL, DIR, ALL, UNIT_STRIDE_0, DIAGONAL
        const std::tuple<int, int, bool, bool, bool> tparams
            = std::make_tuple(twl, dir, all, unit_stride0, diagonal);

        try
        {
            hipLaunchKernelGGL(tmap.at(tparams),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               A,
                               B,
                               (T*)twiddles_large,
                               lengths,
                               stride_in,
                               stride_out);
        }
        catch(std::exception& e)
        {
            rocfft_cout << "scheme: " << scheme << std::endl;
            rocfft_cout << "twl: " << twl << std::endl;
            rocfft_cout << "dir: " << dir << std::endl;
            rocfft_cout << "all: " << all << std::endl;
            rocfft_cout << "diagonal: " << diagonal << std::endl;
            rocfft_cout << e.what() << '\n';
        }
    }
    else
    {
        // Create a map from the parameters to the templated function
        std::map<std::tuple<bool, bool, bool>, // ALL, UNIT_STRIDE_0, DIAGONAL
                 decltype(&HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                                    TA,
                                                                    TB,
                                                                    TRANSPOSE_DIM_X,
                                                                    TRANSPOSE_DIM_Y,
                                                                    true,
                                                                    true,
                                                                    true>))>
            tmap;

        // Fill the map with explicitly instantiated templates:
        tmap.emplace(std::make_tuple(true, true, true),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               true,
                                                               true,
                                                               true>));
        tmap.emplace(std::make_tuple(false, true, true),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               false,
                                                               true,
                                                               true>));
        tmap.emplace(std::make_tuple(true, false, true),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               true,
                                                               false,
                                                               true>));
        tmap.emplace(std::make_tuple(true, true, false),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               true,
                                                               true,
                                                               false>));

        tmap.emplace(std::make_tuple(true, false, false),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               true,
                                                               false,
                                                               false>));

        tmap.emplace(std::make_tuple(false, false, true),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               false,
                                                               false,
                                                               true>));

        tmap.emplace(std::make_tuple(false, true, false),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               false,
                                                               true,
                                                               false>));

        tmap.emplace(std::make_tuple(false, false, false),
                     &HIP_KERNEL_NAME(transpose_kernel2_scheme<T,
                                                               TA,
                                                               TB,
                                                               TRANSPOSE_DIM_X,
                                                               TRANSPOSE_DIM_Y,
                                                               false,
                                                               false,
                                                               false>));

        // Tuple containing template parameters for transpose ALL, UNIT_STRIDE_0, DIAGONAL
        const std::tuple<bool, bool, bool> tparams = std::make_tuple(all, unit_stride0, diagonal);

        try
        {
            hipLaunchKernelGGL(tmap.at(tparams),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               A,
                               B,
                               (T*)twiddles_large,
                               lengths,
                               stride_in,
                               stride_out,
                               ld_in,
                               ld_out,
                               m,
                               n);
        }
        catch(std::exception& e)
        {
            rocfft_cout << "scheme: " << scheme << std::endl;
            rocfft_cout << "twl: " << twl << std::endl;
            rocfft_cout << "dir: " << dir << std::endl;
            rocfft_cout << "all: " << all << std::endl;
            rocfft_cout << "diagonal: " << diagonal << std::endl;
            rocfft_cout << e.what() << '\n';
        }
    }

    return rocfft_status_success;
}

void rocfft_internal_transpose_var2(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t m = data->node->length[1];
    size_t n = data->node->length[0];

    int scheme = 0;
    if(data->node->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
    {
        scheme = 1;
        m      = data->node->length[2];
        n      = data->node->length[0] * data->node->length[1];
    }
    else if(data->node->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
    {
        scheme = 2;
        m      = data->node->length[1] * data->node->length[2];
        n      = data->node->length[0];
    }

    size_t ld_in  = scheme == 1 ? data->node->inStride[2] : data->node->inStride[1];
    size_t ld_out = scheme == 1 ? data->node->outStride[1] : data->node->outStride[2];

    // TODO:
    //   - Might open this option to upstream
    //   - Check it for non-unit stride and other cases
    //   - It seems diagonal transpose is not friendly to XY_Z pow-of-2 cases. Need more investigation.
    bool diagonal = false;

    // NB:
    //    Need better arch comparison other than strings.
    //    Need to check it for Navi and other oncoming new Archs.
    std::string arch(data->deviceProp.gcnArchName);
    if(arch.compare(0, 6, "gfx908") == 0 || arch.compare(0, 6, "gfx90a") == 0)
    {
        diagonal = (m % 256) == 0 && (data->node->outStride[1] % 256 == 0)
                   && (data->node->scheme != CS_KERNEL_TRANSPOSE_XY_Z);
    }
    else
    {
        diagonal = (m % 256) == 0 && (data->node->outStride[1] % 256 == 0);
    }

    // size_t ld_in = data->node->inStride[1];
    // size_t ld_out = data->node->outStride[1];

    // if (ld_in < m )
    //     return rocfft_status_invalid_dimensions;
    // else if (ld_out < n )
    //     return rocfft_status_invalid_dimensions;

    // if(m == 0 || n == 0 ) return rocfft_status_success;

    int twl = 0;

    if(data->node->large1D > (size_t)256 * 256 * 256 * 256)
        printf("large1D twiddle size too large error");
    else if(data->node->large1D > (size_t)256 * 256 * 256)
        twl = 4;
    else if(data->node->large1D > (size_t)256 * 256)
        twl = 3;
    else if(data->node->large1D > (size_t)256)
        twl = 2;
    else if(data->node->large1D > 0)
        twl = 1;
    else
        twl = 0;

    int dir = data->node->direction;

    size_t count = data->node->batch;

    size_t extraDimStart = 2;
    if(scheme != 0)
        extraDimStart = 3;

    hipStream_t rocfft_stream = data->rocfft_stream;

    bool unit_stride0
        = (data->node->inStride[0] == 1 && data->node->outStride[0] == 1) ? true : false;

    for(size_t i = extraDimStart; i < data->node->length.size(); i++)
        count *= data->node->length[i];

    // double2 must use 32 otherwise exceed the shared memory (LDS) size

    // FIXME: push planar ptr on device in better way!!!
    if((data->node->inArrayType == rocfft_array_type_complex_planar
        || data->node->inArrayType == rocfft_array_type_hermitian_planar)
       && (data->node->outArrayType == rocfft_array_type_complex_interleaved
           || data->node->outArrayType == rocfft_array_type_hermitian_interleaved))
    {
        if(data->node->precision == rocfft_precision_single)
        {
            cmplx_float_planar in_planar;
            in_planar.R = (real_type_t<float2>*)data->bufIn[0];
            in_planar.I = (real_type_t<float2>*)data->bufIn[1];

            void* d_in_planar;
            hipMalloc(&d_in_planar, sizeof(cmplx_float_planar));
            hipMemcpy(d_in_planar, &in_planar, sizeof(cmplx_float_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_float,
                                                 cmplx_float_planar,
                                                 cmplx_float,
                                                 64,
                                                 16>(
                m,
                n,
                (const cmplx_float_planar*)d_in_planar,
                (cmplx_float*)data->bufOut[0],
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_in_planar);
        }
        else
        {
            cmplx_double_planar in_planar;
            in_planar.R = (real_type_t<double2>*)data->bufIn[0];
            in_planar.I = (real_type_t<double2>*)data->bufIn[1];

            void* d_in_planar;
            hipMalloc(&d_in_planar, sizeof(cmplx_double_planar));
            hipMemcpy(d_in_planar, &in_planar, sizeof(cmplx_double_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_double,
                                                 cmplx_double_planar,
                                                 cmplx_double,
                                                 32,
                                                 32>(
                m,
                n,
                (const cmplx_double_planar*)d_in_planar,
                (double2*)data->bufOut[0],
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_in_planar);
        }
    }
    else if((data->node->inArrayType == rocfft_array_type_complex_interleaved
             || data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
            && (data->node->outArrayType == rocfft_array_type_complex_planar
                || data->node->outArrayType == rocfft_array_type_hermitian_planar))
    {
        if(data->node->precision == rocfft_precision_single)
        {
            cmplx_float_planar out_planar;
            out_planar.R = (real_type_t<float2>*)data->bufOut[0];
            out_planar.I = (real_type_t<float2>*)data->bufOut[1];

            void* d_out_planar;
            hipMalloc(&d_out_planar, sizeof(cmplx_float_planar));
            hipMemcpy(d_out_planar, &out_planar, sizeof(cmplx_float_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_float,
                                                 cmplx_float,
                                                 cmplx_float_planar,
                                                 64,
                                                 16>(
                m,
                n,
                (const cmplx_float*)data->bufIn[0],
                (cmplx_float_planar*)d_out_planar,
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_out_planar);
        }
        else
        {
            cmplx_double_planar out_planar;
            out_planar.R = (real_type_t<double2>*)data->bufOut[0];
            out_planar.I = (real_type_t<double2>*)data->bufOut[1];

            void* d_out_planar;
            hipMalloc(&d_out_planar, sizeof(cmplx_double_planar));
            hipMemcpy(
                d_out_planar, &out_planar, sizeof(cmplx_double_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_double,
                                                 cmplx_double,
                                                 cmplx_double_planar,
                                                 32,
                                                 32>(
                m,
                n,
                (const cmplx_double*)data->bufIn[0],
                (cmplx_double_planar*)d_out_planar,
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_out_planar);
        }
    }
    else if((data->node->inArrayType == rocfft_array_type_complex_planar
             || data->node->inArrayType == rocfft_array_type_hermitian_planar)
            && (data->node->outArrayType == rocfft_array_type_complex_planar
                || data->node->outArrayType == rocfft_array_type_hermitian_planar))
    {
        if(data->node->precision == rocfft_precision_single)
        {
            cmplx_float_planar in_planar;
            in_planar.R = (real_type_t<float2>*)data->bufIn[0];
            in_planar.I = (real_type_t<float2>*)data->bufIn[1];
            cmplx_float_planar out_planar;
            out_planar.R = (real_type_t<float2>*)data->bufOut[0];
            out_planar.I = (real_type_t<float2>*)data->bufOut[1];

            void* d_in_planar;
            hipMalloc(&d_in_planar, sizeof(cmplx_float_planar));
            hipMemcpy(d_in_planar, &in_planar, sizeof(cmplx_float_planar), hipMemcpyHostToDevice);

            void* d_out_planar;
            hipMalloc(&d_out_planar, sizeof(cmplx_float_planar));
            hipMemcpy(d_out_planar, &out_planar, sizeof(cmplx_float_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_float,
                                                 cmplx_float_planar,
                                                 cmplx_float_planar,
                                                 64,
                                                 16>(
                m,
                n,
                (const cmplx_float_planar*)d_in_planar,
                (cmplx_float_planar*)d_out_planar,
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_in_planar);
            hipFree(d_out_planar);
        }
        else
        {
            cmplx_double_planar in_planar;
            in_planar.R = (real_type_t<double2>*)data->bufIn[0];
            in_planar.I = (real_type_t<double2>*)data->bufIn[1];
            cmplx_double_planar out_planar;
            out_planar.R = (real_type_t<double2>*)data->bufOut[0];
            out_planar.I = (real_type_t<double2>*)data->bufOut[1];

            void* d_in_planar;
            hipMalloc(&d_in_planar, sizeof(cmplx_double_planar));
            hipMemcpy(d_in_planar, &in_planar, sizeof(cmplx_double_planar), hipMemcpyHostToDevice);

            void* d_out_planar;
            hipMalloc(&d_out_planar, sizeof(cmplx_double_planar));
            hipMemcpy(
                d_out_planar, &out_planar, sizeof(cmplx_double_planar), hipMemcpyHostToDevice);

            rocfft_transpose_outofplace_template<cmplx_double,
                                                 cmplx_double_planar,
                                                 cmplx_double_planar,
                                                 32,
                                                 32>(
                m,
                n,
                (const cmplx_double_planar*)(&in_planar),
                (cmplx_double_planar*)(&out_planar),
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);

            hipFree(d_in_planar);
            hipFree(d_out_planar);
        }
    }
    else
    {
        //FIXME:
        //  there are more cases than
        //      if(data->node->inArrayType == rocfft_array_type_complex_interleaved
        //      && data->node->outArrayType == rocfft_array_type_complex_interleaved)
        //  fall into this default case which might to correct
        if(data->node->precision == rocfft_precision_single)
            rocfft_transpose_outofplace_template<cmplx_float, cmplx_float, cmplx_float, 64, 16>(
                m,
                n,
                (const cmplx_float*)data->bufIn[0],
                (cmplx_float*)data->bufOut[0],
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);
        else
            rocfft_transpose_outofplace_template<cmplx_double, cmplx_double, cmplx_double, 32, 32>(
                m,
                n,
                (const cmplx_double*)data->bufIn[0],
                (cmplx_double*)data->bufOut[0],
                data->node->twiddles_large.data(),
                count,
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH,
                twl,
                dir,
                scheme,
                unit_stride0,
                diagonal,
                ld_in,
                ld_out,
                rocfft_stream);
    }
}
