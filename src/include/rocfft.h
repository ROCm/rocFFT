#ifndef __ROCFFT_H__
#define __ROCFFT_H__


#ifdef __HIPCC__
#include <hip_runtime.h>
#endif // __HIPCC__


#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

// Opaque pointers to library internal data structures 
typedef struct rocfft_plan_t *rocfft_plan;
typedef struct rocfft_buffer_t *rocfft_buffer;
typedef struct rocfft_description_t *rocfft_description;
typedef struct rocfft_execution_info_t *rocfft_execution_info;

// Status & error message
typedef enum rocfft_status_t
{
	ROCFFT_STATUS_SUCCESS,
	ROCFFT_STATUS_FAILURE,
} rocfft_status;

// Type of transform
typedef enum rocfft_transform_type_t
{
	ROCFFT_TRANSFORM_TYPE_COMPLEX_FORWARD,
	ROCFFT_TRANSFORM_TYPE_COMPLEX_INVERSE,
	ROCFFT_TRANSFORM_TYPE_REAL_FORWARD,
	ROCFFT_TRANSFORM_TYPE_REAL_INVERSE,	
} rocfft_transform_type;

// Precision
typedef enum rocfft_precision_t
{
	ROCFFT_PRECISION_SINGLE,
	ROCFFT_PRECISION_DOUBLE,
} rocfft_precision;

// Element type
typedef enum rocfft_element_type_t
{
	ROCFFT_ELEMENT_TYPE_COMPLEX_SINGLE,
	ROCFFT_ELEMENT_TYPE_COMPLEX_DOUBLE,
	ROCFFT_ELEMENT_TYPE_SINGLE,
	ROCFFT_ELEMENT_TYPE_DOUBLE,
	ROCFFT_ELEMENT_TYPE_BYTE,	
} rocfft_element_type;

// Result placement
typedef enum rocfft_result_placement_t
{
	ROCFFT_PLACEMENT_INPLACE,
	ROCFFT_PLACEMENT_NOTINPLACE,	
} rocfft_result_placement;

// Array type
typedef enum rocfft_array_type_t
{
	ROCFFT_ARRAY_TYPE_COMPLEX_INTERLEAVED,
	ROCFFT_ARRAY_TYPE_COMPLEX_PLANAR,
	ROCFFT_ARRAY_TYPE_REAL,
	ROCFFT_ARRAY_TYPE_HERMITIAN_INTERLEAVED,
	ROCFFT_ARRAY_TYPE_HERMITIAN_PLANAR,	
} rocfft_array_type;

// Execution mode
typedef enum rocfft_execution_mode_t
{
	ROCFFT_EXEC_MODE_NONBLOCKING,
	ROCFFT_EXEC_MODE_NONBLOCKING_WITH_FLUSH,
	ROCFFT_EXEC_MODE_BLOCKING,
} rocfft_execution_mode;

// library specific malloc and free routines to create device buffers
rocfft_status rocfft_malloc( rocfft_buffer *buffer, rocfft_element_type element_type, size_t size_in_elements );
rocfft_status rocfft_free( rocfft_buffer buffer );


// plan creation in a single step
rocfft_status rocfft_plan_create(	rocfft_plan *plan,
					rocfft_transform_type transform_type, rocfft_precision precision,
					size_t dimensions, const size_t *lengths, size_t number_of_transforms,
					const rocfft_description description );


// plan execution
rocfft_status rocfft_execute(	const rocfft_plan plan,
				rocfft_buffer *in_buffer,
				rocfft_buffer *out_buffer,
				rocfft_execution_info info );

// plan destruction
rocfft_status rocfft_plan_destroy( rocfft_plan plan );

// plan description funtions to specify optional additional plan properties

rocfft_status rocfft_description_set_scale_float( rocfft_description description, float scale );
rocfft_status rocfft_description_set_scale_double( rocfft_description description, double scale );

rocfft_status rocfft_description_set_data_outline(	rocfft_description description,
							rocfft_result_placement placement,
							rocfft_array_type in_array_type, rocfft_array_type out_array_type,
							const size_t *in_offsets, const size_t *out_offsets );

rocfft_status rocfft_description_set_data_layout(	rocfft_description description,
							const size_t *in_strides, size_t in_distance,
							const size_t *out_strides, size_t out_distance );


// get plan information
rocfft_status rocfft_plan_get_work_buffer_size( const rocfft_plan plan, size_t *size_in_bytes );

// execution info set/get functions to control execution and retrieve event/other information
rocfft_status rocfft_execution_info_set_work_buffer( rocfft_execution_info info, rocfft_buffer work_buffer );
rocfft_status rocfft_execution_info_set_mode( rocfft_execution_info info, rocfft_execution_mode mode );


// functions to create and destroy description and execution_info objects 
rocfft_status rocfft_description_create( rocfft_description *description );
rocfft_status rocfft_description_destroy( rocfft_description description );
rocfft_status rocfft_execution_info_create( rocfft_execution_info *info );
rocfft_status rocfft_execution_info_destroy( rocfft_execution_info info );

// print plan details
rocfft_status rocfft_print_plan(const rocfft_plan plan);

// setup function
rocfft_status rocfft_setup();

// cleanup function
rocfft_status rocfft_cleanup();


// HIP exposing functions
#ifdef __HIPCC__

// create buffer, use hip allocated memory space
rocfft_status rocfft_hip_mem_create( rocfft_buffer *buffer, void *p );

// retrieve raw pointer from buffer
rocfft_status rocfft_hip_mem_get_ptr( rocfft_buffer buffer, void **p );

// plan description funtions to specify optional additional plan properties
rocfft_status rocfft_hip_description_set_device( rocfft_description description, int device );

// execution info set/get functions to control execution and retrieve event/other information
rocfft_status rocfft_hip_execution_info_set_stream( rocfft_execution_info info, hipStream_t stream );
rocfft_status rocfft_hip_execution_info_get_events( const rocfft_execution_info info, hipEvent_t *events, size_t number_of_events );

#endif // __HIPCC__


#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_H__
			
					

					
				  

