// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ARRAY_FORMAT_H
#define ARRAY_FORMAT_H

#include "../../../../shared/gpubuf.h"
#include "callback.h"
#include "common.h"
#include "rocfft.h"

//-----------------------------------------------------------------------------
// To support planar format with template, we have the below simple conventions.

template <typename PRECISION>
struct planar
{
    real_type_t<PRECISION>* R; // points to real part array
    real_type_t<PRECISION>* I; // points to imag part array
};

// the default interleaved format
using cmplx_float  = float2;
using cmplx_double = double2;

// the planar format
using cmplx_float_planar  = planar<float2>;
using cmplx_double_planar = planar<double2>;

template <class T>
struct cmplx_type;

template <>
struct cmplx_type<cmplx_float>
{
    typedef float2 type;
};

template <>
struct cmplx_type<double2>
{
    typedef double2 type;
};

template <>
struct cmplx_type<cmplx_float_planar>
{
    typedef float2 type;
};

template <>
struct cmplx_type<cmplx_double_planar>
{
    typedef double2 type;
};

template <class T>
using cmplx_type_t = typename cmplx_type<T>::type;

template <typename T, CallbackType cbtype>
struct Handler
{
};

template <CallbackType cbtype>
struct Handler<cmplx_float, cbtype>
{
    static __host__ __device__ inline float2
        read(cmplx_float const* in, size_t idx, void* load_cb_fn, void* load_cb_data)
    {
        auto load_cb = get_load_cb<float2, cbtype>(load_cb_fn);
        // callback might modify input, but it's otherwise const
        return load_cb(const_cast<cmplx_float*>(in), idx, load_cb_data, nullptr);
    }

    static __host__ __device__ inline void
        write(cmplx_float* out, size_t idx, float2 v, void* store_cb_fn, void* store_cb_data)
    {
        auto store_cb = get_store_cb<float2, cbtype>(store_cb_fn);
        store_cb(out, idx, v, store_cb_data, nullptr);
    }
};

template <CallbackType cbtype>
struct Handler<cmplx_double, cbtype>
{
    static __host__ __device__ inline double2
        read(cmplx_double const* in, size_t idx, void* load_cb_fn, void* load_cb_data)
    {
        auto load_cb = get_load_cb<double2, cbtype>(load_cb_fn);
        // callback might modify input, but it's otherwise const
        return load_cb(const_cast<cmplx_double*>(in), idx, load_cb_data, nullptr);
    }

    static __host__ __device__ inline void
        write(cmplx_double* out, size_t idx, double2 v, void* store_cb_fn, void* store_cb_data)
    {
        auto store_cb = get_store_cb<double2, cbtype>(store_cb_fn);
        store_cb(out, idx, v, store_cb_data, nullptr);
    }
};

template <CallbackType cbtype>
struct Handler<cmplx_float_planar, cbtype>
{
    static __host__ __device__ inline float2
        read(cmplx_float_planar const* in, size_t idx, void* load_cb_fn, void* load_cb_data)
    {
        float2 t;
        t.x = in->R[idx];
        t.y = in->I[idx];
        return t;
    }

    static __host__ __device__ inline void
        write(cmplx_float_planar* out, size_t idx, float2 v, void* store_cb_fn, void* store_cb_data)
    {
        out->R[idx] = v.x;
        out->I[idx] = v.y;
    }
};

template <CallbackType cbtype>
struct Handler<cmplx_double_planar, cbtype>
{
    static __host__ __device__ inline double2
        read(cmplx_double_planar const* in, size_t idx, void* load_cb_fn, void* load_cb_data)
    {
        double2 t;
        t.x = in->R[idx];
        t.y = in->I[idx];
        return t;
    }

    static __host__ __device__ inline void write(
        cmplx_double_planar* out, size_t idx, double2 v, void* store_cb_fn, void* store_cb_data)
    {
        out->R[idx] = v.x;
        out->I[idx] = v.y;
    }
};

static bool is_complex_planar(rocfft_array_type type)
{
    return type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar;
}
static bool is_complex_interleaved(rocfft_array_type type)
{
    return type == rocfft_array_type_complex_interleaved
           || type == rocfft_array_type_hermitian_interleaved;
}

template <class T>
struct cmplx_planar_device_buffer
{
    cmplx_planar_device_buffer(void* real, void* imag)
    {
        planar<T> hostBuf;
        hostBuf.R = static_cast<real_type_t<T>*>(real);
        hostBuf.I = static_cast<real_type_t<T>*>(imag);
        if(deviceBuf.alloc(sizeof(hostBuf)) != hipSuccess)
            throw std::bad_alloc();
        if(hipMemcpy(deviceBuf.data(), &hostBuf, sizeof(hostBuf), hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to copy planar buffer");
    }
    // if we're given const pointers, cheat and cast away const to
    // simplify this struct.  the goal of this struct is to
    // automatically manage the memory, not provide
    // const-correctness.
    cmplx_planar_device_buffer(const void* real, const void* imag)
        : cmplx_planar_device_buffer(const_cast<void*>(real), const_cast<void*>(imag))
    {
    }

    planar<T>* devicePtr()
    {
        return deviceBuf.data();
    }

private:
    gpubuf_t<planar<T>> deviceBuf;
};

#endif
