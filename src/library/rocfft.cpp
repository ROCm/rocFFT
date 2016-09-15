
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













