#define __HIPCC__

#if defined(__NVCC__)
#include "helper_math.h"
#endif

#include <hip_runtime.h>
#include "rocfft_transpose.h"

struct rocfft_transpose_plan_t
{
    size_t rank;
    size_t lengths[3];
    size_t batch;

    rocfft_transpose_precision precision;
    rocfft_transpose_array_type array_type;
    rocfft_transpose_placement placement;
};


struct rocfft_transpose_buffer_t
{
	bool deviceAlloc;
	size_t elementSize;
	void *p;
};

rocfft_transpose_status rocfft_transpose_buffer_create_with_ptr( rocfft_transpose_buffer *buffer, void *p )
{
	rocfft_transpose_buffer b = new rocfft_transpose_buffer_t;
	b->p = p;
	b->deviceAlloc = false;
	b->elementSize =  1;
	*buffer = b;

	return rocfft_transpose_status_success;	
}

rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision, rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,
                                                                 size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                                                 const rocfft_transpose_description *description )
{
        return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             rocfft_transpose_buffer *in_buffer,
                                                             rocfft_transpose_buffer *out_buffer,
                                                             rocfft_transpose_execution_info info )
{
        return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan )
{
        return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_buffer_destroy( rocfft_transpose_buffer buffer )
{
	if(buffer->deviceAlloc)
		hipFree(buffer->p);

	delete buffer;

	return rocfft_transpose_status_success;
}
