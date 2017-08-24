#ifndef __ROCFFT_TRANSPOSE_H__
#define __ROCFFT_TRANSPOSE_H__

#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


rocfft_status
rocfft_transpose_complex_to_complex(rocfft_precision precision, size_t m, size_t n, 
                                    const void* A, size_t lda, 
                                    void* B, size_t ldb, 
                                    size_t batch_count);

#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_TRANSPOSE_H__

