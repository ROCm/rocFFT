/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef COMMON_H
#define COMMON_H
#include "rocfft.h"
#include <hip/hip_vector_types.h>
#include <iostream>

#ifdef __NVCC__
#include "vector_types.h"

__device__ inline float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ inline float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ inline float2 operator*(const float& a, const float2& b)
{
    return make_float2(a * b.x, a * b.y);
}

__device__ inline double2 operator-(const double2& a, const double2& b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ inline double2 operator+(const double2& a, const double2& b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ inline double2 operator*(const double& a, const double2& b)
{
    return make_double2(a * b.x, a * b.y);
}

#endif

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

template <class T>
struct real_type;

template <>
struct real_type<float4>
{
    typedef float type;
};

template <>
struct real_type<double4>
{
    typedef double type;
};

template <>
struct real_type<float2>
{
    typedef float type;
};

template <>
struct real_type<double2>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

/* example of using real_type_t */
// real_type_t<float2> float_scalar;
// real_type_t<double2> double_scalar;

template <class T>
struct vector4_type;

template <>
struct vector4_type<float2>
{
    typedef float4 type;
};

template <>
struct vector4_type<double2>
{
    typedef double4 type;
};

template <class T>
using vector4_type_t = typename vector4_type<T>::type;

/* example of using vector4_type_t */
// vector4_type_t<float2> float4_scalar;
// vector4_type_t<double2> double4_scalar;

template <rocfft_precision T>
struct vector2_type;

template <>
struct vector2_type<rocfft_precision_single>
{
    typedef float2 type;
};

template <>
struct vector2_type<rocfft_precision_double>
{
    typedef double2 type;
};

template <rocfft_precision T>
using vector2_type_t = typename vector2_type<T>::type;

/* example of using vector2_type_t */
// vector2_type_t<rocfft_precision_single> float2_scalar;
// vector2_type_t<rocfft_precision_double> double2_scalar;

template <typename T>
__device__ inline T lib_make_vector2(real_type_t<T> v0, real_type_t<T> v1);

template <>
__device__ inline float2 lib_make_vector2(float v0, float v1)
#ifdef __NVCC__
{
    return make_float2(v0, v1);
}
#else
{
    return float2(v0, v1);
}
#endif

template <>
__device__ inline double2 lib_make_vector2(double v0, double v1)
#ifdef __NVCC__
{
    return make_double2(v0, v1);
}
#else
{
    return double2(v0, v1);
}
#endif

template <typename T>
__device__ inline T
    lib_make_vector4(real_type_t<T> v0, real_type_t<T> v1, real_type_t<T> v2, real_type_t<T> v3);

template <>
__device__ inline float4 lib_make_vector4(float v0, float v1, float v2, float v3)
#ifdef __NVCC__
{
    return make_float4(v0, v1, v2, v3);
}
#else
{
    return float4(v0, v1, v2, v3);
}
#endif

template <>
__device__ inline double4 lib_make_vector4(double v0, double v1, double v2, double v3)
#ifdef __NVCC__
{
    return make_double4(v0, v1, v2, v3);
}
#else
{
    return double4(v0, v1, v2, v3);
}
#endif

template <typename T>
__device__ T TWLstep1(T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    return result;
}

template <typename T>
__device__ T TWLstep2(T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep3(T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep4(T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[768 + j].x - result.y * twiddles[768 + j].y),
                                 (result.y * twiddles[768 + j].x + result.x * twiddles[768 + j].y));
    return result;
}

#define TWIDDLE_STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T     W = TWFUNC(TWIDDLES, INDEX);                 \
        float TR, TI;                                      \
        TR    = (W.x * REG.x) - (W.y * REG.y);             \
        TI    = (W.y * REG.x) + (W.x * REG.y);             \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T     W = TWFUNC(TWIDDLES, INDEX);                 \
        float TR, TI;                                      \
        TR    = (W.x * REG.x) + (W.y * REG.y);             \
        TI    = -(W.y * REG.x) + (W.x * REG.y);            \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#endif // COMMON_H
