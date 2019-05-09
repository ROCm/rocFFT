/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef KERNEL_LAUNCH_SINGLE
#define KERNEL_LAUNCH_SINGLE

#define FN_PRFX(X) rocfft_internal_##X

#include <hip/hip_runtime.h>
#include <vector>

extern "C" {

typedef enum rocfft_transform_type_e
{
    rocfft_transform_type_complex_forward,
    rocfft_transform_type_complex_inverse,
    rocfft_transform_type_real_forward,
    rocfft_transform_type_real_inverse,
} rocfft_transform_type;

// Precision
typedef enum rocfft_precision_e
{
    rocfft_precision_single,
    rocfft_precision_double,
} rocfft_precision;

// Result placement
typedef enum rocfft_result_placement_e
{
    rocfft_placement_inplace,
    rocfft_placement_notinplace,
} rocfft_result_placement;

// Array type
typedef enum rocfft_array_type_e
{
    rocfft_array_type_complex_interleaved,
    rocfft_array_type_complex_planar,
    rocfft_array_type_real,
    rocfft_array_type_hermitian_interleaved,
    rocfft_array_type_hermitian_planar,
} rocfft_array_type;

// Execution mode
typedef enum rocfft_execution_mode_e
{
    rocfft_exec_mode_nonblocking,
    rocfft_exec_mode_nonblocking_with_flush,
    rocfft_exec_mode_blocking,
} rocfft_execution_mode;
}

struct GridParam
{
    size_t b_x, b_y, b_z;
    size_t tpb_x, tpb_y, tpb_z;

    GridParam()
        : b_x(1)
        , b_y(1)
        , b_z(1)
        , tpb_x(1)
        , tpb_y(1)
        , tpb_z(1)
    {
    }
};

enum OperatingBuffer
{
    OB_UNINIT,
    OB_USER_IN,
    OB_USER_OUT,
    OB_TEMP
};

enum ComputeScheme
{
    CS_NONE,
    CS_KERNEL_STOCKHAM,
    CS_KERNEL_STOCKHAM_BLOCK_CC,
    CS_KERNEL_STOCKHAM_BLOCK_RC,
    CS_KERNEL_TRANSPOSE,
    CS_KERNEL_TRANSPOSE_XY_Z,
    CS_KERNEL_TRANSPOSE_Z_XY,
    CS_L1D_TRTRT,
    CS_L1D_CC,
    CS_L1D_CRT,
    CS_2D_STRAIGHT,
    CS_2D_RTRT,
    CS_2D_RC,
    CS_KERNEL_2D_STOCKHAM_BLOCK_CC,
    CS_KERNEL_2D_SINGLE,
    CS_3D_STRAIGHT,
    CS_3D_RTRT,
    CS_3D_RC,
    CS_KERNEL_3D_STOCKHAM_BLOCK_CC,
    CS_KERNEL_3D_SINGLE
};

enum TransTileDir
{
    TTD_IP_HOR,
    TTD_IP_VER,
};

extern "C" {
typedef void (*DevFnCall)(void*, void*);
}

struct DeviceCallIn
{
    size_t batch;

    // transform dimension - note this can be different from data dimension
    size_t dimension;

    // length of the FFT in each dimension
    std::vector<size_t> length;

    // stride of the FFT in each dimension
    std::vector<size_t> inStride, outStride;

    // distance between consecutive batch members
    size_t iDist, oDist;

    size_t                  direction;
    rocfft_result_placement placement;
    rocfft_precision        precision;
    rocfft_array_type       inArrayType, outArrayType;

    // extra twiddle multiplication for large 1D
    size_t large1D;

    ComputeScheme scheme;

    TransTileDir transTileDir;

    void* twiddles;
    void* twiddles_large;

    void* bufIn[2];
    void* bufOut[2];

    GridParam gridParam;
    DevFnCall devFnCall;
};

struct DeviceCallOut
{
    int err;
};

extern "C" {

/* Naming convention

dfn – device function caller (just a prefix, though actually GPU kernel
function)

sp (dp) – single (double) precision

ip – in-place

op - out-of-place

ci – complex-interleaved (format of input buffer)

ci – complex-interleaved (format of output buffer)

stoc – stockham fft kernel

1(2) – one (two) dimension data

1024 – length of fft

*/

void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_4096)(void* data_p, void* back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_2048)(void* data_p, void* back_p);

void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_4096)(void* data_p, void* back_p);
void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_2048)(void* data_p, void* back_p);

void FN_PRFX(transpose_var1_sp)(void* data_p, void* back_p);
}

#endif // KERNEL_LAUNCH_SINGLE
