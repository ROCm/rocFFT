#ifndef __ROCFFT_TRANSPOSE_H__
#define __ROCFFT_TRANSPOSE_H__

#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

// Opaque pointer types to library internal data structures 
typedef struct rocfft_transpose_plan_t *rocfft_transpose_plan;
typedef struct rocfft_transpose_buffer_t *rocfft_transpose_buffer;
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

// Element type
typedef enum rocfft_transpose_element_type_e
{
        rocfft_transpose_element_type_complex_single,
        rocfft_transpose_element_type_complex_double,
        rocfft_transpose_element_type_single,
        rocfft_transpose_element_type_double,
        rocfft_transpose_element_type_byte,        
} rocfft_transpose_element_type;

// Result placement
typedef enum rocfft_transpose_placement_e
{
        rocfft_transpose_placement_inplace,
        rocfft_transpose_placement_notinplace        
} rocfft_transpose_placement;

// Array type
typedef enum rocfft_transpose_array_type_e
{
        rocfft_transpose_array_type_complex_interleaved,
        rocfft_transpose_array_type_complex_planar,
        rocfft_transpose_array_type_real,
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



// library setup function, called once in program at the start of library use
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_setup();

// library cleanup function, called once in program after end of library use
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_cleanup();


// library specific malloc and free routines to create device buffers
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_buffer_create_with_alloc( rocfft_transpose_buffer *buffer, rocfft_transpose_element_type element_type, size_t size_in_elements );
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_buffer_destroy( rocfft_transpose_buffer buffer );

// create buffer, use device memory space already allocated
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_buffer_create_with_ptr( rocfft_transpose_buffer *buffer, void *p );

// retrieve raw device pointer from buffer
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_buffer_get_ptr( rocfft_transpose_buffer buffer, void **p );


// plan creation in a single step
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision, rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,
                                                                 size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                                                 const rocfft_transpose_description *description );


// plan execution
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             rocfft_transpose_buffer *in_buffer,
                                                             rocfft_transpose_buffer *out_buffer,
                                                             rocfft_transpose_execution_info info );

// plan destruction
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan );


// plan description funtions to specify optional additional plan properties
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_description_set_scale_float( rocfft_transpose_description *description, float scale );
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_description_set_scale_double( rocfft_transpose_description *description, double scale );

DLL_PUBLIC rocfft_transpose_status rocfft_transpose_description_set_devices( rocfft_transpose_description description, void *devices, size_t number_of_devices );


// get plan information
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_get_work_buffer_size( const rocfft_transpose_plan plan, size_t *size_in_bytes );


// functions to create and destroy execution_info objects 
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_execution_info_create( rocfft_transpose_execution_info *info );
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_execution_info_destroy( rocfft_transpose_execution_info info );

// execution info set/get functions to control execution and retrieve event/other information
DLL_PUBLIC rocfft_transpose_status rocfft_execution_info_set_work_buffer( rocfft_transpose_execution_info info, rocfft_transpose_buffer work_buffer );
DLL_PUBLIC rocfft_transpose_status rocfft_execution_info_set_mode( rocfft_transpose_execution_info info, rocfft_transpose_execution_mode mode );
DLL_PUBLIC rocfft_transpose_status rocfft_execution_info_set_stream( rocfft_transpose_execution_info info, void *stream );

DLL_PUBLIC rocfft_transpose_status rocfft_execution_info_get_events( const rocfft_transpose_execution_info info, void **events, size_t number_of_events );



#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_TRANSPOSE_H__

