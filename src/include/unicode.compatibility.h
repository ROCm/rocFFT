/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined( amd_unicode_h )
#define amd_unicode_h

//	Typedefs to support unicode and ansii compilation
#if defined( _UNICODE )
	typedef std::wstring		tstring;
	typedef std::wstringstream	tstringstream;
	typedef std::wifstream		tifstream;
	typedef std::wofstream		tofstream;
	typedef std::wfstream		tfstream;
	static std::wostream&	tout	= std::wcout;
	static std::wostream&	terr	= std::wcerr;
#else
	typedef std::string tstring;
	typedef std::stringstream tstringstream;
	typedef std::ifstream		tifstream;
	typedef std::ofstream		tofstream;
	typedef std::fstream		tfstream;
	static std::ostream&	tout	= std::cout;
	static std::ostream&	terr	= std::cerr;
#endif

//	These macros help linux cope with the conventions of windows tchar.h file
#if defined( _WIN32 )
	#include <tchar.h>
	#include <windows.h>
#else
	#if defined( __GNUC__ )
		typedef char TCHAR;
		typedef char _TCHAR;
		#define _tmain main

		#if defined( UNICODE )
			#define _T(x)	L ## x
		#else
			#define _T(x)	x
		#endif
	#endif
#endif

#endif
