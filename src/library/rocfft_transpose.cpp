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


rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision, rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,
                                                                 size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                                                 const rocfft_transpose_description *description )
{
        return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             void *in_buffer,
                                                             void *out_buffer,
                                                             rocfft_transpose_execution_info info )
{
        return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan )
{
        return rocfft_transpose_status_not_implemented;
}
