
#define __HIPCC__

#if defined(__NVCC__)
#include "helper_math.h"
#endif

#include <hip_runtime.h>
#include "rocfft.h"
#include "./plan.h"


// library setup function, called once in program at the start of library use
rocfft_status rocfft_setup()
{
	return rocfft_status_success;
}

// library cleanup function, called once in program after end of library use
rocfft_status rocfft_cleanup()
{
	return rocfft_status_success;
}




rocfft_status rocfft_execution_info_create( rocfft_execution_info *info )
{
	return rocfft_status_success;
}

rocfft_status rocfft_execution_info_destroy( rocfft_execution_info info )
{
	return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size( const rocfft_plan plan, size_t *size_in_bytes )
{
	return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_work_buffer( rocfft_execution_info info, void* work_buffer )
{
	return rocfft_status_success;
}






