
#define __HIPCC__

#include <hip_runtime.h>
#include "rocfft.h"

rocfft_status rocfft_execute(   const rocfft_plan plan,
                                void **in_buffer,
                                void **out_buffer,
                                rocfft_execution_info info )
{
	return rocfft_status_success;
}

