
#ifndef PLAN_H
#define PLAN_H

struct rocfft_plan_description_t
{

	rocfft_array_type inArrayType, outArrayType;

	size_t inStrides[4];
	size_t outStrides[4];

	size_t inOffset[2];
	size_t outOffset[2];

	double scale;
};

struct rocfft_plan_t
{
	size_t rank;
	size_t lengths[3];
	size_t batch;

	rocfft_result_placement	placement;
	rocfft_transform_type	transformType;
	rocfft_precision	precision;

	rocfft_plan_description_t desc;

	rocfft_plan_t() :
		placement(rocfft_placement_inplace),
		rank(1),
		batch(1),
		transformType(rocfft_transform_type_complex_forward),
		precision(rocfft_precision_single)
	{
		lengths[0] = 1;
		lengths[1] = 1;
		lengths[2] = 1;		
	}	
};



#endif // PLAN_H


