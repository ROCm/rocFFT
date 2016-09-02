
#include <iostream>
#include <sstream>
#include <functional>
#include <cmath>

#include <boost/program_options.hpp>
#include "rocfft.h"
#include "../../src/include/unicode.compatibility.h"

namespace po = boost::program_options;

//	This is used with the program_options class so that the user can type an integer on the command line
//	and we store into an enum varaible
template<class _Elem, class _Traits>
std::basic_istream<_Elem, _Traits> & operator>> (std::basic_istream<_Elem, _Traits> & stream, rocfft_array_type & atype)
{
	unsigned tmp;
	stream >> tmp;
	atype = rocfft_array_type(tmp);
	return stream;
}

// similarly for transform type
template<class _Elem, class _Traits>
std::basic_istream<_Elem, _Traits> & operator>> (std::basic_istream<_Elem, _Traits> & stream, rocfft_transform_type & ttype)
{
	unsigned tmp;
	stream >> tmp;
	ttype = rocfft_transform_type(tmp);
	return stream;
}


int _tmain( int argc, _TCHAR* argv[] )
{
	//	This helps with mixing output of both wide and narrow characters to the screen
	std::ios::sync_with_stdio( false );

	int				deviceId = 0;
	int				platformId = 0;

	//	FFT state

	rocfft_result_placement place = rocfft_placement_inplace;
	rocfft_transform_type	transformType = rocfft_transform_type_complex_forward;
	rocfft_array_type	inArrType  = rocfft_array_type_complex_interleaved;
	rocfft_array_type	outArrType = rocfft_array_type_complex_interleaved;
	rocfft_precision precision = rocfft_precision_single;

	size_t lengths[ 3 ] = {1,1,1};
	size_t iStrides[ 4 ] = {0,0,0,0};
	size_t oStrides[ 4 ] = {0,0,0,0};
	unsigned profile_count = 0;

	unsigned command_queue_flags = 0;
	size_t batchSize = 1;

	try
	{
		// Declare the supported options.
		po::options_description desc( "rocfft rider command line options" );
		desc.add_options()
			( "help,h",        "produces this help message" )
			( "version,v",     "Print queryable version information from the rocfft library" )
			( "info,i",      "Print queryable information of all the runtimes and devices" )
			( "printChosen",   "Print queryable information of the selected runtime and device" )
			( "platform",      po::value< int >( &platformId )->default_value( 0 ),   "Select a specific platform id as it is reported by info" )
			( "device",        po::value< int >( &deviceId )->default_value( 0 ),   "Select a specific device id as it is reported by info" )
			( "notInPlace,o",    "Not in-place FFT transform (default: in-place)" )
			( "double",		   "Double precision transform (default: single)" )
			( "transformType,t",	po::value< rocfft_transform_type >( &transformType )->default_value( rocfft_transform_type_complex_forward ), "Type of transform:\n0) complex forward\n1) complex inverse\n2) real forward\n3) real inverse" )
			( "lenX,x",        po::value< size_t >( &lengths[ 0 ] )->default_value( 1024 ),   "Specify the length of the 1st dimension of a test array" )
			( "lenY,y",        po::value< size_t >( &lengths[ 1 ] )->default_value( 1 ),      "Specify the length of the 2nd dimension of a test array" )
			( "lenZ,z",        po::value< size_t >( &lengths[ 2 ] )->default_value( 1 ),      "Specify the length of the 3rd dimension of a test array" )
			( "isX",   po::value< size_t >( &iStrides[ 0 ] )->default_value( 1 ),		"Specify the input stride of the 1st dimension of a test array" )
			( "isY",   po::value< size_t >( &iStrides[ 1 ] )->default_value( 0 ),	"Specify the input stride of the 2nd dimension of a test array" )
			( "isZ",   po::value< size_t >( &iStrides[ 2 ] )->default_value( 0 ),	"Specify the input stride of the 3rd dimension of a test array" )
			( "iD", po::value< size_t >( &iStrides[ 3 ] )->default_value( 0 ), "input distance between successive members when batch size > 1" )
			( "osX",   po::value< size_t >( &oStrides[ 0 ] )->default_value( 1 ),		"Specify the output stride of the 1st dimension of a test array" )
			( "osY",   po::value< size_t >( &oStrides[ 1 ] )->default_value( 0 ),	"Specify the output stride of the 2nd dimension of a test array" )
			( "osZ",   po::value< size_t >( &oStrides[ 2 ] )->default_value( 0 ),	"Specify the output stride of the 3rd dimension of a test array" )
			( "oD", po::value< size_t >( &oStrides[ 3 ] )->default_value( 0 ), "output distance between successive members when batch size > 1" )
			( "batchSize,b",   po::value< size_t >( &batchSize )->default_value( 1 ), "If this value is greater than one, arrays will be used " )
			( "profile,p",     po::value< unsigned >( &profile_count )->default_value( 1 ), "Time and report the kernel speed of the FFT (default: profiling off)" )
			( "inArrType",      po::value< rocfft_array_type >( &inArrType )->default_value( rocfft_array_type_complex_interleaved ), "Array type of input data:\n0) interleaved\n1) planar\n2) hermitian interleaved\n3) hermitian planar\n4) real" )
			( "outArrType",     po::value< rocfft_array_type >( &outArrType )->default_value( rocfft_array_type_complex_interleaved ), "Array type of output data:\n0) interleaved\n1) planar\n2) hermitian interleaved\n3) hermitian planar\n4) real" )
			;

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		if( vm.count( "version" ) )
		{
			std::cout << "version" << std::endl;
			return 0;
		}

		if( vm.count( "help" ) )
		{
			//	This needs to be 'cout' as program-options does not support wcout yet
			std::cout << desc << std::endl;
			return 0;
		}

		
		if( vm.count( "info" ) )
		{
			return 0;
		}

		bool printInfo = false;
		if( vm.count( "printChosen" ) )
		{
			printInfo = true;
		}

		if( vm.count( "notInPlace" ) )
		{
			place = rocfft_placement_notinplace;
		}

		if( vm.count( "double" ) )
		{
			precision = rocfft_precision_double;
		}


		if( profile_count > 1 )
		{
		}

		int inL = (int)inArrType;
		int otL = (int)outArrType;

		// input output array type support matrix
		int ioArrTypeSupport[5][5] =		{
										{ 1, 1, 0, 0, 1 },
										{ 1, 1, 0, 0, 1 },
										{ 0, 0, 0, 0, 1 },
										{ 0, 0, 0, 0, 1 },
										{ 1, 1, 1, 1, 0 },
										};

		if(inL > 4) throw std::runtime_error( "Invalid Input array type format" );
		if(otL > 4) throw std::runtime_error( "Invalid Output array type format" );

		if(ioArrTypeSupport[inL][otL] == 0) throw std::runtime_error( "Invalid combination of Input/Output array type formats" );

		if( (transformType == rocfft_transform_type_complex_forward) || (transformType == rocfft_transform_type_complex_inverse) ) // Complex-Complex cases
		{
			iStrides[1] = iStrides[1] ? iStrides[1] : lengths[0] * iStrides[0];
			iStrides[2] = iStrides[2] ? iStrides[2] : lengths[1] * iStrides[1];
			iStrides[3] = iStrides[3] ? iStrides[3] : lengths[2] * iStrides[2];

			if(place == rocfft_placement_inplace)
			{
				oStrides[0] = iStrides[0];
				oStrides[1] = iStrides[1];
				oStrides[2] = iStrides[2];
				oStrides[3] = iStrides[3];
			}
			else
			{
				oStrides[1] = oStrides[1] ? oStrides[1] : lengths[0] * oStrides[0];
				oStrides[2] = oStrides[2] ? oStrides[2] : lengths[1] * oStrides[1];
				oStrides[3] = oStrides[3] ? oStrides[3] : lengths[2] * oStrides[2];
			}
		}
		else // Real cases
		{
			size_t *rst, *cst;
			size_t N = lengths[0];
			size_t Nt = 1 + lengths[0]/2;
			bool iflag = false;
			bool rcFull = (inL == 0) || (inL == 1) || (otL == 0) || (otL == 1);

			if(inArrType == rocfft_array_type_real ) { iflag = true; rst = iStrides; }
			else { rst = oStrides; } // either in or out should be REAL

			// Set either in or out strides whichever is real
			if(place == rocfft_placement_inplace)
			{
				if(rcFull)	{ rst[1] = rst[1] ? rst[1] :  N * 2 * rst[0]; }
				else		{ rst[1] = rst[1] ? rst[1] : Nt * 2 * rst[0]; }

				rst[2] = rst[2] ? rst[2] : lengths[1] * rst[1];
				rst[3] = rst[3] ? rst[3] : lengths[2] * rst[2];
			}
			else
			{
				rst[1] = rst[1] ? rst[1] : lengths[0] * rst[0];
				rst[2] = rst[2] ? rst[2] : lengths[1] * rst[1];
				rst[3] = rst[3] ? rst[3] : lengths[2] * rst[2];
			}

			// Set the remaining of in or out strides that is not real
			if(iflag) { cst = oStrides; }
			else	  { cst = iStrides; }

			if(rcFull)	{ cst[1] = cst[1] ? cst[1] :  N * cst[0]; }
			else		{ cst[1] = cst[1] ? cst[1] : Nt * cst[0]; }

			cst[2] = cst[2] ? cst[2] : lengths[1] * cst[1];
			cst[3] = cst[3] ? cst[3] : lengths[2] * cst[2];
		}
		/*
		if( precision == CLFFT_SINGLE )
			transform<float>( lengths, iStrides, oStrides, batchSize, inArrType, outArrType, place, precision, dir, deviceType, deviceId, platformId, printInfo, command_queue_flags, profile_count, setupData );
		else
			transform<double>( lengths, iStrides, oStrides, batchSize, inArrType, outArrType, place, precision, dir, deviceType, deviceId, platformId, printInfo, command_queue_flags, profile_count, setupData ); */
	}
	catch( std::exception& e )
	{
		terr << _T( "rocfft error condition reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}

	return 0;
}

