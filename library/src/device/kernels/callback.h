// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_DEVICE_CALLBACK_H
#define ROCFFT_DEVICE_CALLBACK_H

#include <hip/hip_vector_types.h>

// user-provided data saying what callbacks to run
struct UserCallbacks
{
    void*  load_cb_fn        = nullptr;
    void*  load_cb_data      = nullptr;
    size_t load_cb_lds_bytes = 0;

    void*  store_cb_fn        = nullptr;
    void*  store_cb_data      = nullptr;
    size_t store_cb_lds_bytes = 0;
};

// default callback implementations that just do simple load/store
template <typename T>
__device__ T load_cb_default(T* data, size_t offset, void* cbdata, void* sharedMem)
{
    return data[offset];
}

template <typename T>
__device__ void store_cb_default(T* data, size_t offset, T element, void* cbdata, void* sharedMem)
{
    data[offset] = element;
}

// callback function types
template <typename T>
struct callback_type;

template <>
struct callback_type<float>
{
    typedef float (*load)(float* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(float* data, size_t offset, float element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_float  = load_cb_default<float>;
static __device__ auto store_cb_default_float = store_cb_default<float>;

template <>
struct callback_type<float2>
{
    typedef float2 (*load)(float2* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        float2* data, size_t offset, float2 element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_float2  = load_cb_default<float2>;
static __device__ auto store_cb_default_float2 = store_cb_default<float2>;

template <>
struct callback_type<double>
{
    typedef double (*load)(double* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        double* data, size_t offset, double element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_double  = load_cb_default<double>;
static __device__ auto store_cb_default_double = store_cb_default<double>;

template <>
struct callback_type<double2>
{
    typedef double2 (*load)(double2* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        double2* data, size_t offset, double2 element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_double2  = load_cb_default<double2>;
static __device__ auto store_cb_default_double2 = store_cb_default<double2>;

enum struct CallbackType
{
    // don't run user callbacks
    NONE,
    // run user load/store callbacks
    USER_LOAD_STORE,
};

// helpers to cast void* to the correct function pointer type
template <typename T, CallbackType cbtype>
static __device__ typename callback_type<T>::load get_load_cb(void* ptr)
{
#ifdef ROCFFT_CALLBACKS_ENABLED
    if(cbtype == CallbackType::USER_LOAD_STORE)
        return reinterpret_cast<typename callback_type<T>::load>(ptr);
#endif
    return load_cb_default<T>;
}

template <typename T, CallbackType cbtype>
static __device__ typename callback_type<T>::store get_store_cb(void* ptr)
{
#ifdef ROCFFT_CALLBACKS_ENABLED
    if(cbtype == CallbackType::USER_LOAD_STORE)
        return reinterpret_cast<typename callback_type<T>::store>(ptr);
#endif
    return store_cb_default<T>;
}

#endif
