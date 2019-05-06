/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(_generator_param_H)
#define _generator_param_H
#include "rocfft.h"

/* =====================================================================
    Parameter used to control kernel generation
   =================================================================== */

enum BlockComputeType
{
    BCT_C2C, // Column to column
    BCT_C2R, // Column to row
    BCT_R2C, // Row to column
};

struct FFTKernelGenKeyParams
{

#define ROCFFT_MAX_INTERNAL_DIM 16
    /*
   *    This structure distills a subset of the fftPlan data,
   *    including all information that is used to generate the FFT kernel.
   *    This structure can be used as a key to reusing kernels that have already
   *    been compiled.
   */
    size_t fft_DataDim; // Dimensionality of the data
    size_t fft_N[ROCFFT_MAX_INTERNAL_DIM]; // [0] is FFT size, e.g. 1024
    // This must be <= size of LDS!
    // size_t                   fft_inStride [ROCFFT_MAX_INTERNAL_DIM];  // input
    // strides, TODO: in rocFFT, stride is a run-time parameter of the kernel
    // size_t                   fft_outStride[ROCFFT_MAX_INTERNAL_DIM];  // output
    // strides TODO: in rocFFT, stride is a run-time parameter of the kernel

    rocfft_array_type fft_inputLayout;
    rocfft_array_type fft_outputLayout;
    rocfft_precision  fft_precision;
    double            fft_fwdScale;
    double            fft_backScale;

    size_t fft_workGroupSize; // Assume this workgroup size
    size_t fft_LDSsize; // Limit the use of LDS to this many bytes.
    size_t fft_numTrans; // # numTransforms in a workgroup

    size_t fft_MaxWorkGroupSize; // Limit for work group size

    bool fft_3StepTwiddle; // This is one pass of the "3-step" algorithm;
    // so extra twiddles are applied on output.
    bool fft_twiddleFront; // do twiddle scaling at the beginning pass

    bool fft_realSpecial; // this is the flag to control the special case step
    // (4th step)
    // in the 5-step real 1D large breakdown
    size_t fft_realSpecial_Nr;

    bool fft_RCsimple;

    bool transOutHorizontal; // tiles traverse the output buffer in horizontal
    // direction

    bool   blockCompute;
    size_t blockSIMD;
    size_t blockLDS;

    BlockComputeType blockComputeType;

    std::string name_suffix; // use to specify kernel & device functions names to
    // avoid naming conflict.
    // sometimes non square matrix are broken down into a number of
    // square matrix during inplace transpose
    // let's call this number transposeMiniBatchSize
    // no user of the library should set its value
    size_t transposeMiniBatchSize;
    // transposeBatchSize is the number of batchs times transposeMiniBatchSzie
    // no user of the library should set its value
    size_t transposeBatchSize;

    bool fft_hasPreCallback; // two call back variable
    bool fft_hasPostCallback;

    long limit_LocalMemSize;

    // Default constructor
    FFTKernelGenKeyParams()
    {
        fft_DataDim = 0;
        for(int i = 0; i < ROCFFT_MAX_INTERNAL_DIM; i++)
        {
            fft_N[i] = 0;
            // fft_inStride[i] = 0;
            // fft_outStride[i] = 0;
        }

        fft_inputLayout  = rocfft_array_type_complex_interleaved;
        fft_outputLayout = rocfft_array_type_complex_interleaved;
        fft_precision    = rocfft_precision_single;
        fft_fwdScale = fft_backScale = 0.0;
        fft_workGroupSize            = 0;
        fft_LDSsize                  = 0;
        fft_numTrans                 = 1;
        fft_MaxWorkGroupSize         = 256;
        fft_3StepTwiddle             = false;
        fft_twiddleFront             = false;

        transOutHorizontal = false;

        fft_realSpecial    = false;
        fft_realSpecial_Nr = 0;

        fft_RCsimple = false;

        blockCompute     = false;
        blockSIMD        = 0;
        blockLDS         = 0;
        blockComputeType = BCT_C2C;

        transposeMiniBatchSize = 1;
        transposeBatchSize     = 1;
        fft_hasPreCallback     = false;
        fft_hasPostCallback    = false;
        limit_LocalMemSize     = 0;
    }
};

#endif // _generator_param_H
