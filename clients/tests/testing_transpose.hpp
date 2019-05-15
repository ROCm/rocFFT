/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "../rider/misc.h" // to use LIB_V_THROW and HIP_V_THROW
#include "rocfft.h"
#include "rocfft_transpose.h"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))

using namespace std;

/* ============================================================================================
 */
/* generate random number :*/
template <typename T>
T random_generator()
{
    return (T)(rand() % 10 + 1); // generate a integer number between [1, 10]
};

/* ============================================================================================
 */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same
// value
template <typename T>
void rocfft_init(vector<T>& A, size_t M, size_t N, size_t lda)
{
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            A[i + j * lda] = random_generator<T>();
        }
    }
};

template <typename T>
void unit_check_general(size_t M, size_t N, size_t lda, T* hCPU, T* hGPU)
{
#pragma unroll
    for(size_t j = 0; j < N; j++)
    {
#pragma unroll
        for(size_t i = 0; i < M; i++)
        {
            EXPECT_EQ((hCPU[i + j * lda]).x, (hGPU[i + j * lda]).x);
            EXPECT_EQ((hCPU[i + j * lda]).y, (hGPU[i + j * lda]).y);
        }
    }
}

template <typename T>
void print_matrix(vector<T> CPU_result, vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
        {
            printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n",
                   i,
                   j,
                   (CPU_result[j + i * lda]).x,
                   (GPU_result[j + i * lda]).x);
            printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n",
                   i,
                   j,
                   (CPU_result[j + i * lda]).y,
                   (GPU_result[j + i * lda]).x);
        }
}

template <typename T>
void transpose_reference(size_t m, size_t n, T* A, size_t lda, T* B, size_t ldb, size_t batch_count)
{
    // transpose per batch
    for(size_t b = 0; b < batch_count; b++)
    {
        for(size_t i = 0; i < m; i++)
        {
#pragma unroll
            for(size_t j = 0; j < n; j++)
            {
                B[b * m * ldb + j + i * ldb] = A[b * n * lda + i + j * lda];
            }
        }
    }
}

template <class T>
rocfft_status
    rocfft_transpose(size_t m, size_t n, T* A, size_t lda, T* B, size_t ldb, size_t batch_count);

template <>
rocfft_status rocfft_transpose<float2>(
    size_t m, size_t n, float2* A, size_t lda, float2* B, size_t ldb, size_t batch_count)
{
    return rocfft_status_success;
    // return rocfft_transpose_complex_to_complex(rocfft_precision_single, m, n,
    // (const void*) A, lda, (void*)B, ldb, batch_count);
}

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/

template <typename T>
rocfft_status testing_transpose(size_t M, size_t N, size_t lda, size_t ldb, size_t batch_count)
{

    T *dA, *dB;

    printf("M=%d, N=%d, lda=%d, ldb=%d\n", M, N, lda, ldb);
    size_t A_size, B_size;

    rocfft_status status;

    A_size = lda * N * batch_count;
    B_size = ldb * M * batch_count;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hB_copy(B_size);

    // allocate memory on device
    HIP_V_THROW(hipMalloc(&dA, A_size * sizeof(T)), "hipMalloc failed");
    HIP_V_THROW(hipMalloc(&dB, B_size * sizeof(T)), "hipMalloc failed");

    // Initial Data on CPU
    srand(1);
    rocfft_init<T>(hA, M, N * batch_count, lda);
    rocfft_init<T>(hB, N, M * batch_count, ldb);

    // copy data from CPU to device, does not work for lda != A_row
    HIP_V_THROW(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice),
                "hipMemcpy failed");

    /* =====================================================================
       rocfft
  =================================================================== */

    // library interface

    status = rocfft_transpose<T>(M, N, dA, lda, dB, ldb, batch_count);

    if(status != rocfft_status_success) // only valid size, compare with cblas
    {
        HIP_V_THROW(hipFree(dA), "hipMalloc failed");
        HIP_V_THROW(hipFree(dB), "hipMalloc failed");
        return status;
    }

    // copy output from device to CPU
    HIP_V_THROW(hipMemcpy(hB.data(), dB, sizeof(T) * B_size, hipMemcpyDeviceToHost),
                "hipMemcpy failed");

    /* =====================================================================
              CPU Implementation
  =================================================================== */
    if(status != rocfft_status_invalid_dimensions) // only valid size, compare with cblas
    {
        transpose_reference<T>(M, N, hA.data(), lda, hB_copy.data(), ldb, batch_count);

        print_matrix(hB_copy, hB, min(N, 3), min(M, 3), ldb);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order

        for(int i = 0; i < batch_count; i++)
        {
            unit_check_general<T>(N, M, ldb, hB_copy.data() + M * ldb * i, hB.data() + M * ldb * i);
        }
    }

    HIP_V_THROW(hipFree(dA), "hipMalloc failed");
    HIP_V_THROW(hipFree(dB), "hipMalloc failed");

    return status;
}
