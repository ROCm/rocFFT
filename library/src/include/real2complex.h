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

#ifndef REAL_TO_COMPLEX_H
#define REAL_TO_COMPLEX_H

// The even-length real to complex post process device kernel
template <typename Tcomplex, bool Ndiv4>
__device__ inline void post_process_interleaved(const size_t    idx_p,
                                                const size_t    idx_q,
                                                const size_t    half_N,
                                                const size_t    quarter_N,
                                                const Tcomplex* input,
                                                Tcomplex*       output,
                                                const Tcomplex* twiddles)
{
    if(idx_p == 0)
    {
        output[half_N].x = input[0].x - input[0].y;
        output[half_N].y = 0;
        output[0].x      = input[0].x + input[0].y;
        output[0].y      = 0;

        if(Ndiv4)
        {
            output[quarter_N].x = input[quarter_N].x;
            output[quarter_N].y = -input[quarter_N].y;
        }
    }
    else
    {
        const Tcomplex p = input[idx_p];
        const Tcomplex q = input[idx_q];
        const Tcomplex u = 0.5 * (p + q);
        const Tcomplex v = 0.5 * (p - q);

        const Tcomplex twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        output[idx_p].x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        output[idx_p].y = v.y + u.y * twd_p.y - v.x * twd_p.x;

        output[idx_q].x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        output[idx_q].y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
    }
}

// TODO: merge back to post_process_interleaved()
template <typename T, bool Ndiv4>
__device__ inline void post_process_interleaved_inplace(const size_t idx_p,
                                                        const size_t idx_q,
                                                        const size_t half_N,
                                                        const size_t quarter_N,
                                                        T*           inout,
                                                        const T*     twiddles)
{
    T p, q;
    if(idx_p < quarter_N)
    {
        p = inout[idx_p];
        q = inout[idx_q];
    }

    __syncthreads();

    if(idx_p == 0)
    {
        inout[idx_p].x = p.x + p.y;
        inout[idx_p].y = 0;
        inout[idx_q].x = p.x - p.y;
        inout[idx_q].y = 0;

        if(Ndiv4)
            inout[quarter_N].y = -inout[quarter_N].y;
    }
    else if(idx_p < quarter_N)
    {
        const T u = 0.5 * (p + q);
        const T v = 0.5 * (p - q);

        const T twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        inout[idx_p].x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        inout[idx_p].y = v.y + u.y * twd_p.y - v.x * twd_p.x;

        inout[idx_q].x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        inout[idx_q].y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
    }
}

void real2complex(const void* data, void* back);
void complex2hermitian(const void* data, void* back);

void hermitian2complex(const void* data, void* back);
void complex2real(const void* data, void* back);

void r2c_1d_post(const void* data, void* back);
void r2c_1d_post_transpose(const void* data, void* back);
void c2r_1d_pre(const void* data, void* back);
void transpose_c2r_1d_pre(const void* data, void* back);

void complex2pair_unpack(const void* data, void* back);
void pair2complex_pack(const void* data, void* back);

#endif // REAL_TO_COMPLEX_H
