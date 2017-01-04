/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include "rocfft.h"

#ifdef __NVCC__

#include "vector_types.h"

__device__ float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x-b.x, a.y-b.y); }
__device__ float2 operator+(const float2 &a, const float2 &b) { return make_float2(a.x+b.x, a.y+b.y); }
__device__ float2 operator*(const float &a, const float2 &b) { return make_float2(a*b.x, a*b.y); }

#endif


template<class T>
struct real_type;

template<>
struct real_type<float2>
{
    typedef float type;
};

template<>
struct real_type<double2>
{
    typedef double type;
};

template<class T>
using real_type_t = typename real_type<T>::type;

/* example of using real_type_t */
//real_type_t<float2> float_scalar;
//real_type_t<double2> double_scalar;

template<class T>
struct vector4_type;

template<>
struct vector4_type<float2>
{
    typedef float4 type;
};

template<>
struct vector4_type<double2>
{
    typedef double4 type;
};

template<class T>
using vector4_type_t = typename vector4_type<T>::type;


/* example of using vector4_type_t */
//vector4_type_t<float2> float4_scalar;
//vector4_type_t<double2> double4_scalar;


template<rocfft_precision T>
struct vector2_type;

template<>
struct vector2_type<rocfft_precision_single>
{
    typedef float2 type;
};

template<>
struct vector2_type<rocfft_precision_double>
{
    typedef double2 type;
};

template<rocfft_precision T>
using vector2_type_t = typename vector2_type<T>::type;


/* example of using vector2_type_t */
//vector2_type_t<rocfft_precision_single> float2_scalar;
//vector2_type_t<rocfft_precision_double> double2_scalar;
        


#ifdef __NVCC__
#define MAKE_FLOAT2 make_float2
#define MAKE_FLOAT4 make_float4
#else
#define MAKE_FLOAT2 float2
#define MAKE_FLOAT4 float4


//TODO: temporary solution may not nvcc compatible
#define MAKE_COMPLEX T

#endif



#endif // COMMON_H

