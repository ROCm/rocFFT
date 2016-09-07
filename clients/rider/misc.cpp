
#define __HIPCC__

#include <hip_runtime.h>
#include "rocfft.h"
#include "./misc.h"

void setupBuffers( std::vector< int > devices,
                     const size_t bufferSizeBytesIn,
                     const unsigned numBuffersIn,
                     void *buffersIn[],
                     const size_t bufferSizeBytesOut,
                     const unsigned numBuffersOut,
                     void *buffersOut[] )
{
	for(unsigned i=0; i<numBuffersIn; i++)
		HIP_V_THROW( hipMalloc(&buffersIn[i], bufferSizeBytesIn), "hipMalloc failed" );

	for(unsigned i=0; i<numBuffersOut; i++)
		HIP_V_THROW( hipMalloc(&buffersOut[i], bufferSizeBytesOut), "hipMalloc failed" );

}

void clearBuffers(   
			const unsigned numBuffersIn,
                	void *buffersIn[],
        	        const unsigned numBuffersOut,
                	void *buffersOut[] )
{

	for(unsigned i=0; i<numBuffersIn; i++)
		HIP_V_THROW( hipFree(buffersIn[i]), "hipFree failed" );

	for(unsigned i=0; i<numBuffersOut; i++)
		HIP_V_THROW( hipFree(buffersOut[i]), "hipFree failed" );

}


