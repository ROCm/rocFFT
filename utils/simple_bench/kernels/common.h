/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef COMMON_H
#define COMMON_H

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

template <typename T>
__device__ inline T lib_make_vector2(real_type_t<T> v0, real_type_t<T> v1);

template <>
__device__ inline float2 lib_make_vector2(float v0, float v1)
{
    return float2(v0, v1);
}

template <>
__device__ inline double2 lib_make_vector2(double v0, double v1)
{
    return double2(v0, v1);
}

template <typename T>
__device__ inline T
    lib_make_vector4(real_type_t<T> v0, real_type_t<T> v1, real_type_t<T> v2, real_type_t<T> v3);

template <>
__device__ inline float4 lib_make_vector4(float v0, float v1, float v2, float v3)
{
    return float4(v0, v1, v2, v3);
}

template <>
__device__ inline double4 lib_make_vector4(double v0, double v1, double v2, double v3)
{
    return double4(v0, v1, v2, v3);
}

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

#endif // COMMON_H
