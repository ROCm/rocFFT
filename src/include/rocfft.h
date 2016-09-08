#ifndef __ROCFFT_H__
#define __ROCFFT_H__

#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

// Opaque pointer types to library internal data structures 
typedef struct rocfft_plan_t *rocfft_plan;
typedef struct rocfft_plan_description_t *rocfft_plan_description;
typedef struct rocfft_execution_info_t *rocfft_execution_info;

// Status & error message
typedef enum rocfft_status_e
{
        rocfft_status_success,
        rocfft_status_failure,
	rocfft_status_invalid_arg_value,
} rocfft_status;

// Type of transform
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



// library setup function, called once in program at the start of library use
DLL_PUBLIC rocfft_status rocfft_setup();

// library cleanup function, called once in program after end of library use
DLL_PUBLIC rocfft_status rocfft_cleanup();


// plan creation in a single step
DLL_PUBLIC rocfft_status rocfft_plan_create(       rocfft_plan *plan,
                                        rocfft_transform_type transform_type, rocfft_precision precision,
                                        size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                        const rocfft_plan_description description );


// plan execution
DLL_PUBLIC rocfft_status rocfft_execute(   const rocfft_plan plan,
                                void **in_buffer,
                                void **out_buffer,
                                rocfft_execution_info info );

// plan destruction
DLL_PUBLIC rocfft_status rocfft_plan_destroy( rocfft_plan plan );


// plan description funtions to specify optional additional plan properties
DLL_PUBLIC rocfft_status rocfft_plan_description_set_scale_float( rocfft_plan_description description, float scale );
DLL_PUBLIC rocfft_status rocfft_plan_description_set_scale_double( rocfft_plan_description description, double scale );

DLL_PUBLIC rocfft_status rocfft_plan_description_set_data_outline(      rocfft_plan_description description,
                                                        rocfft_result_placement placement,
                                                        rocfft_array_type in_array_type, rocfft_array_type out_array_type,
                                                        const size_t *in_offsets, const size_t *out_offsets );

DLL_PUBLIC rocfft_status rocfft_plan_description_set_data_layout(       rocfft_plan_description description,
                                                        const size_t *in_strides, size_t in_distance,
                                                        const size_t *out_strides, size_t out_distance );

DLL_PUBLIC rocfft_status rocfft_plan_description_set_devices( rocfft_plan_description description, void *devices, size_t number_of_devices );


// get plan information
DLL_PUBLIC rocfft_status rocfft_plan_get_work_buffer_size( const rocfft_plan plan, size_t *size_in_bytes );


// functions to create and destroy description and execution_info objects
DLL_PUBLIC rocfft_status rocfft_plan_description_create( rocfft_plan_description *description );
DLL_PUBLIC rocfft_status rocfft_plan_description_destroy( rocfft_plan_description description );
DLL_PUBLIC rocfft_status rocfft_execution_info_create( rocfft_execution_info *info );
DLL_PUBLIC rocfft_status rocfft_execution_info_destroy( rocfft_execution_info info );

// execution info set/get functions to control execution and retrieve event/other information
DLL_PUBLIC rocfft_status rocfft_execution_info_set_work_buffer( rocfft_execution_info info, void* work_buffer );
DLL_PUBLIC rocfft_status rocfft_execution_info_set_mode( rocfft_execution_info info, rocfft_execution_mode mode );
DLL_PUBLIC rocfft_status rocfft_execution_info_set_stream( rocfft_execution_info info, void *stream );

DLL_PUBLIC rocfft_status rocfft_execution_info_get_events( const rocfft_execution_info info, void **events, size_t number_of_events );



#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_H__

