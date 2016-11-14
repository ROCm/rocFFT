/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <sstream>
#include <vector>
#include <hip/hip_runtime_api.h>

#define countOf( arr ) ( sizeof( arr ) / sizeof( arr[ 0 ] ) )

void setupBuffers( std::vector< int > devices,
                     const size_t bufferSizeBytesIn,
                     const unsigned numBuffersIn,
                     void *buffersIn[],
                     const size_t bufferSizeBytesOut,
                     const unsigned numBuffersOut,
                     void *buffersOut[] );

void clearBuffers(
                	void *buffersIn[],
                	void *buffersOut[] );


//	This is used to either wrap a HIP function call, or to explicitly check a variable for an OpenCL error condition.
//	If an error occurs, we throw.
//	Note: std::runtime_error does not take unicode strings as input, so only strings supported
inline hipError_t hip_V_Throw ( hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName )
{
	switch( res )
	{
		case	hipSuccess:		/**< No error */
			break;
		default:
		{
			std::stringstream tmp;
			tmp << "HIP_V_THROWERROR< ";
			//tmp << prettyPrintclFFTStatus( res );
			tmp << res;
			tmp << " > (";
			tmp << fileName;
			tmp << " Line: ";
			tmp << lineno;
			tmp << "): ";
			tmp << msg;
			std::string errorm (tmp.str());
			std::cout << errorm<< std::endl;
			throw	std::runtime_error( errorm );
		}
	}

	return	res;
}

inline rocfft_status lib_V_Throw ( rocfft_status res, const std::string& msg, size_t lineno, const std::string& fileName )
{
	switch( res )
	{
		case	rocfft_status_success:		/**< No error */
			break;
		default:
		{
			std::stringstream tmp;
			tmp << "LIB_V_THROWERROR< ";
			//tmp << prettyPrintclFFTStatus( res );
			tmp << res;
			tmp << " > (";
			tmp << fileName;
			tmp << " Line: ";
			tmp << lineno;
			tmp << "): ";
			tmp << msg;
			std::string errorm (tmp.str());
			std::cout << errorm<< std::endl;
			throw	std::runtime_error( errorm );
		}
	}

	return	res;
}

#define HIP_V_THROW(_status,_message) hip_V_Throw (_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status,_message) lib_V_Throw (_status, _message, __LINE__, __FILE__)

#endif // MISC_H


