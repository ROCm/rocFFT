#ifndef __ROCFFT_TRANSPOSE_H__
#define __ROCFFT_TRANSPOSE_H__

#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

// Opaque pointer types to library internal data structures 
typedef struct rocfft_transpose_plan_t *rocfft_transpose_plan;
typedef struct rocfft_transpose_description_t *rocfft_transpose_description;
typedef struct rocfft_transpose_execution_info_t *rocfft_transpose_execution_info;

// Status & error message
typedef enum rocfft_transpose_status_e
{
        rocfft_transpose_status_success,
        rocfft_transpose_status_failure,
        rocfft_transpose_status_not_implemented
} rocfft_transpose_status;


// Precision
typedef enum rocfft_transpose_precision_e
{
        rocfft_transpose_precision_single,
        rocfft_transpose_precision_double,
} rocfft_transpose_precision;


// Result placement
typedef enum rocfft_transpose_placement_e
{
        rocfft_transpose_placement_inplace,
        rocfft_transpose_placement_notinplace        
} rocfft_transpose_placement;

// Array type
typedef enum rocfft_transpose_array_type_e
{
        rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved,
        rocfft_transpose_array_type_complex_interleaved_to_complex_planar,
        rocfft_transpose_array_type_complex_planar_to_complex_planar,
        rocfft_transpose_array_type_complex_planar_to_complex_interleaved,
        rocfft_transpose_array_type_real_to_real,
        rocfft_transpose_array_type_hermitian_interleaved,
        rocfft_transpose_array_type_hermitian_planar,        
} rocfft_transpose_array_type;

// Execution mode
typedef enum rocfft_transpose_execution_mode_e
{
        rocfft_transpose_exec_mode_nonblocking,
        rocfft_transpose_exec_mode_nonblocking_with_flush,
        rocfft_transpose_exec_mode_blocking,
} rocfft_transpose_execution_mode;

// plan creation in a single step
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision, 
                                                                 rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,// inplace or outplace
                                                                 size_t dimensions, // dimension of input and output matrix
                                                                 const size_t *lengths, // sizes for each dimension of input matrix
                                                                 const size_t *in_stride, // input matrix stride
                                                                 const size_t *out_stride, // output matrix stide
                                                                 const size_t in_dist, // input distance between batches
                                                                 const size_t out_dist, // output distance between batches
                                                                 size_t number_of_transforms, // number of batches
                                                                 const rocfft_transpose_description *description );


// plan execution
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             void **in_buffer,
                                                             void **out_buffer,
                                                             rocfft_transpose_execution_info info );

// plan destruction
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan );

#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_TRANSPOSE_H__

