#include <vector>
#include <assert.h>
#include <iostream>

#include "rocfft.h"
#include "./plan.h"


size_t Large1DThreshold = 4096;

static inline bool IsPo2(size_t u) {
	return (u != 0) && (0 == (u & (u - 1)));
}

inline size_t PrecisionWidth(rocfft_precision pr)
{
	switch (pr)
	{
	case rocfft_precision_single:	return 1;
	case rocfft_precision_double:	return 2;
	default:		assert(false);	return 1;
	}
}



rocfft_status rocfft_plan_description_set_scale_float( rocfft_plan_description description, float scale )
{
	description->scale = scale;

	return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_scale_double( rocfft_plan_description description, double scale )
{
	description->scale = scale;

	return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_data_layout(       rocfft_plan_description description,
                                                        rocfft_array_type in_array_type, rocfft_array_type out_array_type,
                                                        const size_t *in_offsets, const size_t *out_offsets,
                                                        size_t in_strides_size, const size_t *in_strides, size_t in_distance,
                                                        size_t out_strides_size, const size_t *out_strides, size_t out_distance )
{
	description->inArrayType = in_array_type;
	description->outArrayType = out_array_type;

	if(in_offsets != nullptr)
	{
		description->inOffset[0] = in_offsets[0];
		if( (in_array_type == rocfft_array_type_complex_planar) || (in_array_type == rocfft_array_type_hermitian_planar) )
			description->inOffset[1] = in_offsets[1];
	}

	if(out_offsets != nullptr)
	{
		description->outOffset[0] = out_offsets[0];
		if( (out_array_type == rocfft_array_type_complex_planar) || (out_array_type == rocfft_array_type_hermitian_planar) )
			description->outOffset[1] = out_offsets[1];
	}


	if(in_strides != nullptr)
	{
		for(size_t i=0; i<MIN(3, in_strides_size); i++)
			description->inStrides[i] = in_strides[i];
	}

	if(in_distance != 0)
		description->inDist = in_distance;

	if(out_strides != nullptr)
	{
		for(size_t i=0; i<MIN(3, out_strides_size); i++)
			description->outStrides[i] = out_strides[i];
	}

	if(out_distance != 0)
		description->outDist = out_distance;


	return rocfft_status_success;
}

rocfft_status rocfft_plan_description_create( rocfft_plan_description *description )
{
	rocfft_plan_description desc = new rocfft_plan_description_t;
	*description = desc;

	return rocfft_status_success;
}

rocfft_status rocfft_plan_description_destroy( rocfft_plan_description description )
{
	if(description != nullptr)
		delete description;

	return rocfft_status_success;
}


rocfft_status rocfft_plan_create(       rocfft_plan *plan,
					rocfft_result_placement placement,
                                        rocfft_transform_type transform_type, rocfft_precision precision,
                                        size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                        const rocfft_plan_description description )
{
	switch(transform_type)
	{
	case rocfft_transform_type_complex_forward:
	case rocfft_transform_type_complex_inverse:
	{
		if(placement == rocfft_placement_inplace)
		{
			if(description->inArrayType == rocfft_array_type_complex_interleaved)
			{
				if(description->outArrayType != rocfft_array_type_complex_interleaved)
					return rocfft_status_invalid_array_type;
			}
			else if(description->inArrayType == rocfft_array_type_complex_planar)
			{
				if(description->outArrayType != rocfft_array_type_complex_planar)
					return rocfft_status_invalid_array_type;
			}
			else
				return rocfft_status_invalid_array_type;
		}
		else
		{
			if( (	(description->inArrayType == rocfft_array_type_complex_interleaved) ||
				(description->inArrayType == rocfft_array_type_complex_planar) ) )
			{
				if( !(	(description->outArrayType == rocfft_array_type_complex_interleaved) ||
					(description->outArrayType == rocfft_array_type_complex_planar) ) )
					return rocfft_status_invalid_array_type;
			}
			else
				return rocfft_status_invalid_array_type;
		}
	}
	break;
	case rocfft_transform_type_real_forward:
	{
	}
	break;
	case rocfft_transform_type_real_inverse:
	{
	}
	break;
	}


	if( (placement == rocfft_placement_inplace) &&
		((transform_type == rocfft_transform_type_complex_forward) || (transform_type == rocfft_transform_type_complex_inverse)) )
	{
		for(size_t i=0; i<3; i++)
			if(description->inStrides[i] != description->outStrides[i])
				return rocfft_status_invalid_strides;

		if(description->inDist != description->outDist)
			return rocfft_status_invalid_distance;

		for(size_t i=0; i<2; i++)
			if(description->inOffset[i] != description->outOffset[i])
				return rocfft_status_invalid_offset;
	}

	if(dimensions > 3)
		return rocfft_status_invalid_dimensions;


	rocfft_plan p = new rocfft_plan_t;
	p->rank = dimensions;

	for(size_t i=0; i<(p->rank); i++)
		p->lengths[i] = lengths[i];

	p->batch = number_of_transforms;
	p->placement = placement;

	if(description != nullptr)
		p->desc = *description;

	if(p->desc.inStrides[0] == 0)
	{
		p->desc.inStrides[0] = 1;
		for(size_t i=1; i<(p->rank); i++)
			p->desc.inStrides[i] = p->lengths[i-1] * p->desc.inStrides[i-1];
	}

	if(p->desc.outStrides[0] == 0)
	{
		p->desc.outStrides[0] = 1;
		for(size_t i=1; i<(p->rank); i++)
			p->desc.outStrides[i] = p->lengths[i-1] * p->desc.outStrides[i-1];
	}


	if(p->desc.inDist == 0)
	{
		p->desc.inDist = 1;
		for(size_t i=0; i<(p->rank); i++)
			p->desc.inDist *= p->lengths[i];
	}

	if(p->desc.outDist == 0)
	{
		p->desc.outDist = 1;
		for(size_t i=0; i<(p->rank); i++)
			p->desc.outDist *= p->lengths[i];
	}


	*plan = p;

	return rocfft_status_success;
}

rocfft_status rocfft_plan_destroy( rocfft_plan plan )
{
	if(plan != nullptr)
		delete plan;

	return rocfft_status_success;
}


enum OperatingBuffer
{
	OB_UNINIT,
	OB_USER_IN,
	OB_USER_OUT,
	OB_TEMP
};

enum ComputeScheme
{
	CS_NONE,
	CS_KERNEL_STOCKHAM,
	CS_KERNEL_STOCKHAM_BLOCK_CC,
	CS_KERNEL_STOCKHAM_BLOCK_RC,
	CS_KERNEL_TRANSPOSE,
	CS_KERNEL_TRANSPOSE_XY_Z,
	CS_KERNEL_TRANSPOSE_Z_XY,
	CS_L1D_TRTRT,
	CS_L1D_CC,
	CS_L1D_CRT,
	CS_2D_STRAIGHT,
	CS_2D_RTRT,
	CS_2D_RC,
	CS_KERNEL_2D_STOCKHAM_BLOCK_CC,
	CS_KERNEL_2D_SINGLE,
	CS_3D_STRAIGHT,
	CS_3D_RTRT,
	CS_3D_RC,
	CS_KERNEL_3D_STOCKHAM_BLOCK_CC,
	CS_KERNEL_3D_SINGLE
};

std::string PrintScheme(ComputeScheme cs)
{
	std::string str;

	switch (cs)
	{
		case CS_KERNEL_STOCKHAM:					str += "CS_KERNEL_STOCKHAM				"; break;
		case CS_KERNEL_STOCKHAM_BLOCK_CC:			str += "CS_KERNEL_STOCKHAM_BLOCK_CC		"; break;
		case CS_KERNEL_STOCKHAM_BLOCK_RC:			str += "CS_KERNEL_STOCKHAM_BLOCK_RC		"; break;
		case CS_KERNEL_TRANSPOSE:					str += "CS_KERNEL_TRANSPOSE				"; break;
		case CS_KERNEL_TRANSPOSE_XY_Z:				str += "CS_KERNEL_TRANSPOSE_XY_Z		"; break;
		case CS_KERNEL_TRANSPOSE_Z_XY:				str += "CS_KERNEL_TRANSPOSE_Z_XY		"; break;
		case CS_L1D_TRTRT:							str += "CS_L1D_TRTRT					"; break;
		case CS_L1D_CC:								str += "CS_L1D_CC						"; break;
		case CS_L1D_CRT:							str += "CS_L1D_CRT						"; break;
		case CS_2D_STRAIGHT:						str += "CS_2D_STRAIGHT					"; break;
		case CS_2D_RTRT:							str += "CS_2D_RTRT						"; break;
		case CS_2D_RC:								str += "CS_2D_RC						"; break;
		case CS_KERNEL_2D_STOCKHAM_BLOCK_CC:		str += "CS_KERNEL_2D_STOCKHAM_BLOCK_CC	"; break;
		case CS_KERNEL_2D_SINGLE:					str += "CS_KERNEL_2D_SINGLE				"; break;
		case CS_3D_STRAIGHT:						str += "CS_3D_STRAIGHT					"; break;
		case CS_3D_RTRT:							str += "CS_3D_RTRT						"; break;
		case CS_3D_RC:								str += "CS_3D_RC						"; break;
		case CS_KERNEL_3D_STOCKHAM_BLOCK_CC:		str += "CS_KERNEL_3D_STOCKHAM_BLOCK_CC	"; break;
		case CS_KERNEL_3D_SINGLE:					str += "CS_KERNEL_3D_SINGLE				"; break;

		default:									str += "CS_NONE							"; break;
	}

	return str;
}



class TreeNode
{
private:
	// disallow public creation
	TreeNode(TreeNode *p) : parent(p), scheme(CS_NONE), obIn(OB_UNINIT), obOut(OB_UNINIT), large1D(0)
	{}

public:
	size_t						batchsize;

	// transform dimension - note this can be different from data dimension
	size_t						dimension;

	// length of the FFT in each dimension
	std::vector< size_t >		length;

	// stride of the FFT in each dimension
	std::vector< size_t >		inStride, outStride;

	// distance between consecutive batch members
	size_t						iDist, oDist;

	rocfft_result_placement	placement;
	rocfft_precision			precision;
	rocfft_array_type			inArrayType, outArrayType;

	// extra twiddle multiplication for large 1D
	size_t						large1D;

	TreeNode					*parent;
	std::vector<TreeNode *>		childNodes; 

	ComputeScheme				scheme;
	OperatingBuffer				obIn, obOut;

public:

	TreeNode(const TreeNode &) = delete;			// disallow copy constructor
	TreeNode& operator=(const TreeNode&) = delete;	// disallow assignment operator

	// create node (user level) using this function
	static TreeNode* CreateNode(TreeNode *parentNode = nullptr)
	{
		return new TreeNode(parentNode);
	}

	// destroy node by calling this function
	static void DeleteNode(TreeNode *node)
	{
		std::vector<TreeNode *>::iterator children_p;
		for (children_p = node->childNodes.begin(); children_p != node->childNodes.end(); children_p++)
			DeleteNode(*children_p); // recursively delete allocated nodes

		delete node;
	}

	void RecursiveBuildTree()
	{
		switch (dimension)
		{
		case 1:
		{
			if (length[0] <= Large1DThreshold)
			{
				scheme = CS_KERNEL_STOCKHAM;
				return;
			}

			size_t divLength1 = 1;

			if (IsPo2(length[0]))
			{
				// Enable block compute under these conditions
				if (length[0] <= 262144 / PrecisionWidth(precision))
				{
					if (1 == PrecisionWidth(precision))
					{
						switch (length[0])
						{
						case 8192:		divLength1 = 64;		break;
						case 16384:		divLength1 = 64;		break;
						case 32768:		divLength1 = 128;		break;
						case 65536:		divLength1 = 256;		break;
						case 131072:	divLength1 = 64;		break;
						case 262144:	divLength1 = 64;		break;
						default:		assert(false);
						}
					}
					else
					{
						switch (length[0])
						{
						case 4096:		divLength1 = 64;		break;
						case 8192:		divLength1 = 64;		break;
						case 16384:		divLength1 = 64;		break;
						case 32768:		divLength1 = 128;		break;
						case 65536:		divLength1 = 64;		break;
						case 131072:	divLength1 = 64;		break;
						default:		assert(false);
						}
					}

					scheme = (length[0] <= 65536 / PrecisionWidth(precision)) ? CS_L1D_CC : CS_L1D_CRT;
				}
				else
				{
					if (length[0] > (Large1DThreshold * Large1DThreshold))
					{
						divLength1 = length[0] / Large1DThreshold;
					}
					else
					{
						size_t in_x = 0;
						size_t len = length[0];

						while (len != 1) { len >>= 1; in_x++; }

						in_x /= 2;
						divLength1 = (size_t)1 << in_x;
					}

					scheme = CS_L1D_TRTRT;
				}
			}
			else
			{
			}

			size_t divLength0 = length[0] / divLength1;

			switch (scheme)
			{
			case CS_L1D_TRTRT:
			{
				// first transpose
				TreeNode *trans1Plan = TreeNode::CreateNode(this);
				trans1Plan->precision = precision;
				trans1Plan->batchsize = batchsize;

				trans1Plan->length.push_back(divLength0);
				trans1Plan->length.push_back(divLength1);

				trans1Plan->scheme = CS_KERNEL_TRANSPOSE;
				trans1Plan->dimension = 2;

				for (size_t index = 1; index < length.size(); index++)
				{
					trans1Plan->length.push_back(length[index]);
				}

				childNodes.push_back(trans1Plan);

				// first row fft
				TreeNode *row1Plan = TreeNode::CreateNode(this);
				row1Plan->precision = precision;
				row1Plan->batchsize = batchsize;

				// twiddling is done in row2 or transpose2
				row1Plan->large1D = 0;

				row1Plan->length.push_back(divLength1);
				row1Plan->length.push_back(divLength0);

				row1Plan->scheme = CS_KERNEL_STOCKHAM;
				row1Plan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row1Plan->length.push_back(length[index]);
				}

				row1Plan->RecursiveBuildTree();
				childNodes.push_back(row1Plan);

				// second transpose
				TreeNode *trans2Plan = TreeNode::CreateNode(this);
				trans2Plan->precision = precision;
				trans2Plan->batchsize = batchsize;

				trans2Plan->length.push_back(divLength1);
				trans2Plan->length.push_back(divLength0);

				trans2Plan->scheme = CS_KERNEL_TRANSPOSE;
				trans2Plan->dimension = 2;

				trans2Plan->large1D = length[0];

				for (size_t index = 1; index < length.size(); index++)
				{
					trans2Plan->length.push_back(length[index]);
				}

				childNodes.push_back(trans2Plan);

				// second row fft
				TreeNode *row2Plan = TreeNode::CreateNode(this);
				row2Plan->precision = precision;
				row2Plan->batchsize = batchsize;

				row2Plan->length.push_back(divLength0);
				row2Plan->length.push_back(divLength1);

				row2Plan->scheme = CS_KERNEL_STOCKHAM;
				row2Plan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row2Plan->length.push_back(length[index]);
				}

				// algorithm is set up in a way that row2 does not recurse
				assert(divLength0 <= Large1DThreshold);

				childNodes.push_back(row2Plan);

				// third transpose
				TreeNode *trans3Plan = TreeNode::CreateNode(this);
				trans3Plan->precision = precision;
				trans3Plan->batchsize = batchsize;

				trans3Plan->length.push_back(divLength0);
				trans3Plan->length.push_back(divLength1);

				trans3Plan->scheme = CS_KERNEL_TRANSPOSE;
				trans3Plan->dimension = 2;

				for (size_t index = 1; index < length.size(); index++)
				{
					trans3Plan->length.push_back(length[index]);
				}			
				
				childNodes.push_back(trans3Plan);
			}
			break;
			case CS_L1D_CC:
			{
				// first plan, column-to-column
				TreeNode *col2colPlan = TreeNode::CreateNode(this);
				col2colPlan->precision = precision;
				col2colPlan->batchsize = batchsize;

				// large1D flag to confirm we need multiply twiddle factor
				col2colPlan->large1D = length[0];

				col2colPlan->length.push_back(divLength1);
				col2colPlan->length.push_back(divLength0);

				col2colPlan->scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
				col2colPlan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					col2colPlan->length.push_back(length[index]);
				}

				childNodes.push_back(col2colPlan);

				// second plan, row-to-column
				TreeNode *row2colPlan = TreeNode::CreateNode(this);
				row2colPlan->precision = precision;
				row2colPlan->batchsize = batchsize;

				row2colPlan->length.push_back(divLength0);
				row2colPlan->length.push_back(divLength1);

				row2colPlan->scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;
				row2colPlan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row2colPlan->length.push_back(length[index]);
				}

				childNodes.push_back(row2colPlan);
			}
			break;
			case CS_L1D_CRT:
			{
				// first plan, column-to-column
				TreeNode *col2colPlan = TreeNode::CreateNode(this);
				col2colPlan->precision = precision;
				col2colPlan->batchsize = batchsize;

				// large1D flag to confirm we need multiply twiddle factor
				col2colPlan->large1D = length[0];

				col2colPlan->length.push_back(divLength1);
				col2colPlan->length.push_back(divLength0);

				col2colPlan->scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
				col2colPlan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					col2colPlan->length.push_back(length[index]);
				}

				childNodes.push_back(col2colPlan);

				// second plan, row-to-row
				TreeNode *row2rowPlan = TreeNode::CreateNode(this);
				row2rowPlan->precision = precision;
				row2rowPlan->batchsize = batchsize;

				row2rowPlan->length.push_back(divLength0);
				row2rowPlan->length.push_back(divLength1);

				row2rowPlan->scheme = CS_KERNEL_STOCKHAM;
				row2rowPlan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row2rowPlan->length.push_back(length[index]);
				}

				childNodes.push_back(row2rowPlan);

				// third plan, transpose
				TreeNode *transPlan = TreeNode::CreateNode(this);
				transPlan->precision = precision;
				transPlan->batchsize = batchsize;

				transPlan->length.push_back(divLength0);
				transPlan->length.push_back(divLength1);

				transPlan->scheme = CS_KERNEL_TRANSPOSE;
				transPlan->dimension = 2;

				for (size_t index = 1; index < length.size(); index++)
				{
					transPlan->length.push_back(length[index]);
				}

				childNodes.push_back(transPlan);
			}
			break;
			default: assert(false);

			}

		}
		break;

		case 2:
		{
			if (scheme == CS_KERNEL_TRANSPOSE)
				return;

			// conditions to choose which scheme
			if ((length[0] * length[1]) <= 2048)
				scheme = CS_KERNEL_2D_SINGLE;
			else if (length[1] <= 256)
				scheme = CS_2D_RC;
			else
				scheme = CS_2D_RTRT;

			switch (scheme)
			{
			case CS_2D_RTRT:
			{
				// first row fft
				TreeNode *row1Plan = TreeNode::CreateNode(this);
				row1Plan->precision = precision;
				row1Plan->batchsize = batchsize;

				row1Plan->length.push_back(length[0]);
				row1Plan->dimension = 1;
				row1Plan->length.push_back(length[1]);

				for (size_t index = 2; index < length.size(); index++)
				{
					row1Plan->length.push_back(length[index]);
				}

				row1Plan->RecursiveBuildTree();
				childNodes.push_back(row1Plan);

				// first transpose
				TreeNode *trans1Plan = TreeNode::CreateNode(this);
				trans1Plan->precision = precision;
				trans1Plan->batchsize = batchsize;

				trans1Plan->length.push_back(length[0]);
				trans1Plan->length.push_back(length[1]);

				trans1Plan->scheme = CS_KERNEL_TRANSPOSE;
				trans1Plan->dimension = 2;

				for (size_t index = 2; index < length.size(); index++)
				{
					trans1Plan->length.push_back(length[index]);
				}

				childNodes.push_back(trans1Plan);

				// second row fft
				TreeNode *row2Plan = TreeNode::CreateNode(this);
				row2Plan->precision = precision;
				row2Plan->batchsize = batchsize;

				row2Plan->length.push_back(length[1]);
				row2Plan->dimension = 1;
				row2Plan->length.push_back(length[0]);

				for (size_t index = 2; index < length.size(); index++)
				{
					row2Plan->length.push_back(length[index]);
				}

				row2Plan->RecursiveBuildTree();
				childNodes.push_back(row2Plan);

				// second transpose
				TreeNode *trans2Plan = TreeNode::CreateNode(this);
				trans2Plan->precision = precision;
				trans2Plan->batchsize = batchsize;

				trans2Plan->length.push_back(length[1]);
				trans2Plan->length.push_back(length[0]);

				trans2Plan->scheme = CS_KERNEL_TRANSPOSE;
				trans2Plan->dimension = 2;

				for (size_t index = 2; index < length.size(); index++)
				{
					trans2Plan->length.push_back(length[index]);
				}
				
				childNodes.push_back(trans2Plan);

			}
			break;
			case CS_2D_RC:
			{
				// row fft
				TreeNode *rowPlan = TreeNode::CreateNode(this);
				rowPlan->precision = precision;
				rowPlan->batchsize = batchsize;

				rowPlan->length.push_back(length[0]);
				rowPlan->dimension = 1;
				rowPlan->length.push_back(length[1]);

				for (size_t index = 2; index < length.size(); index++)
				{
					rowPlan->length.push_back(length[index]);
				}

				rowPlan->RecursiveBuildTree();
				childNodes.push_back(rowPlan);

				// column fft
				TreeNode *colPlan = TreeNode::CreateNode(this);
				colPlan->precision = precision;
				colPlan->batchsize = batchsize;

				colPlan->length.push_back(length[1]);
				colPlan->dimension = 1;
				colPlan->length.push_back(length[0]);

				for (size_t index = 2; index < length.size(); index++)
				{
					colPlan->length.push_back(length[index]);
				}

				colPlan->scheme = CS_KERNEL_2D_STOCKHAM_BLOCK_CC;
				childNodes.push_back(colPlan);

			}
			break;
			case CS_KERNEL_2D_SINGLE:
			{

			}
			break;

			default: assert(false);
			}

		}		
		break;

		case 3:
		{
			// conditions to choose which scheme
			if ((length[0] * length[1] * length[2]) <= 2048)
				scheme = CS_KERNEL_3D_SINGLE;
			else if (length[2] <= 256)
				scheme = CS_3D_RC;
			else
				scheme = CS_3D_RTRT;

			switch (scheme)
			{
			case CS_3D_RTRT:
			{
				// 2d fft
				TreeNode *xyPlan = TreeNode::CreateNode(this);
				xyPlan->precision = precision;
				xyPlan->batchsize = batchsize;

				xyPlan->length.push_back(length[0]);
				xyPlan->length.push_back(length[1]);
				xyPlan->dimension = 2;
				xyPlan->length.push_back(length[2]);

				for (size_t index = 3; index < length.size(); index++)
				{
					xyPlan->length.push_back(length[index]);
				}

				xyPlan->RecursiveBuildTree();
				childNodes.push_back(xyPlan);

				// first transpose
				TreeNode *trans1Plan = TreeNode::CreateNode(this);
				trans1Plan->precision = precision;
				trans1Plan->batchsize = batchsize;

				trans1Plan->length.push_back(length[0]);
				trans1Plan->length.push_back(length[1]);
				trans1Plan->length.push_back(length[2]);

				trans1Plan->scheme = CS_KERNEL_TRANSPOSE_XY_Z;
				trans1Plan->dimension = 2;

				for (size_t index = 3; index < length.size(); index++)
				{
					trans1Plan->length.push_back(length[index]);
				}

				childNodes.push_back(trans1Plan);

				// z fft
				TreeNode *zPlan = TreeNode::CreateNode(this);
				zPlan->precision = precision;
				zPlan->batchsize = batchsize;

				zPlan->length.push_back(length[2]);
				zPlan->dimension = 1;
				zPlan->length.push_back(length[0]);
				zPlan->length.push_back(length[1]);

				for (size_t index = 3; index < length.size(); index++)
				{
					zPlan->length.push_back(length[index]);
				}

				zPlan->RecursiveBuildTree();
				childNodes.push_back(zPlan);

				// second transpose
				TreeNode *trans2Plan = TreeNode::CreateNode(this);
				trans2Plan->precision = precision;
				trans2Plan->batchsize = batchsize;

				trans2Plan->length.push_back(length[2]);
				trans2Plan->length.push_back(length[0]);
				trans2Plan->length.push_back(length[1]);

				trans2Plan->scheme = CS_KERNEL_TRANSPOSE_Z_XY;
				trans2Plan->dimension = 2;

				for (size_t index = 3; index < length.size(); index++)
				{
					trans2Plan->length.push_back(length[index]);
				}

				childNodes.push_back(trans2Plan);
			}
			break;
			case CS_3D_RC:
			{
				// 2d fft
				TreeNode *xyPlan = TreeNode::CreateNode(this);
				xyPlan->precision = precision;
				xyPlan->batchsize = batchsize;

				xyPlan->length.push_back(length[0]);
				xyPlan->length.push_back(length[1]);
				xyPlan->dimension = 2;
				xyPlan->length.push_back(length[2]);

				for (size_t index = 3; index < length.size(); index++)
				{
					xyPlan->length.push_back(length[index]);
				}

				xyPlan->RecursiveBuildTree();
				childNodes.push_back(xyPlan);

				// z col fft
				TreeNode *zPlan = TreeNode::CreateNode(this);
				zPlan->precision = precision;
				zPlan->batchsize = batchsize;

				zPlan->length.push_back(length[2]);
				zPlan->dimension = 1;
				zPlan->length.push_back(length[0]);
				zPlan->length.push_back(length[1]);

				for (size_t index = 3; index < length.size(); index++)
				{
					zPlan->length.push_back(length[index]);
				}

				zPlan->scheme = CS_KERNEL_3D_STOCKHAM_BLOCK_CC;
				childNodes.push_back(zPlan);

			}
			break;
			case CS_KERNEL_3D_SINGLE:
			{

			}
			break;

			default: assert(false);
			}
		}
		break;

		default: assert(false);
		}

	}

	// logic A - using out-of-place transposes & complex-to-complex & with padding 
	void TraverseTreeAssignBuffersLogicA(OperatingBuffer &flipIn, OperatingBuffer &flipOut)
	{
		if (parent == nullptr)
		{
			flipIn = OB_USER_OUT;
			flipOut = OB_TEMP;
		}

		if (scheme == CS_L1D_TRTRT)
		{
			if (parent == nullptr)
			{
				childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
			}
			else
				childNodes[0]->obIn = flipIn;

			childNodes[0]->obOut = flipOut;

			OperatingBuffer t;
			t = flipIn;
			flipIn = flipOut;
			flipOut = t;

			if (childNodes[1]->childNodes.size())
			{
				childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);

				size_t cs = childNodes[1]->childNodes.size();
				childNodes[1]->obIn = childNodes[1]->childNodes[0]->obIn;
				childNodes[1]->obOut = childNodes[1]->childNodes[cs-1]->obOut;
			}
			else
			{
				childNodes[1]->obIn = flipIn;
				childNodes[1]->obOut = OB_USER_OUT;

				if (flipIn != OB_USER_OUT)
				{
					OperatingBuffer t;
					t = flipIn;
					flipIn = flipOut;
					flipOut = t;
				}
			}

			if ((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
			{
				if (flipIn == OB_TEMP)
				{
					childNodes[2]->obIn = OB_TEMP;
					childNodes[2]->obOut = OB_USER_OUT;

					childNodes[3]->obIn = OB_USER_OUT;
					childNodes[3]->obOut = OB_TEMP;

					childNodes[4]->obIn = OB_TEMP;
					childNodes[4]->obOut = OB_USER_OUT;
				}
				else
				{
					childNodes[2]->obIn = OB_USER_OUT;
					childNodes[2]->obOut = OB_TEMP;

					childNodes[3]->obIn = OB_TEMP;
					childNodes[3]->obOut = OB_TEMP;

					childNodes[4]->obIn = OB_TEMP;
					childNodes[4]->obOut = OB_USER_OUT;
				}

				obIn = childNodes[0]->obIn;
				obOut = childNodes[4]->obOut;
			}
			else
			{
				assert(obIn == obOut);
				
				if (obOut == OB_USER_OUT)
				{
					if (childNodes[1]->obOut == OB_TEMP)
					{
						childNodes[2]->obIn = OB_TEMP;
						childNodes[2]->obOut = OB_USER_OUT;

						childNodes[3]->obIn = OB_USER_OUT;
						childNodes[3]->obOut = OB_TEMP;

						childNodes[4]->obIn = OB_TEMP;
						childNodes[4]->obOut = OB_USER_OUT;
					}
					else
					{
						childNodes[2]->obIn = OB_USER_OUT;
						childNodes[2]->obOut = OB_TEMP;

						childNodes[3]->obIn = OB_TEMP;
						childNodes[3]->obOut = OB_TEMP;

						childNodes[4]->obIn = OB_TEMP;
						childNodes[4]->obOut = OB_USER_OUT;
					}
				}
				else
				{
					if (childNodes[1]->obOut == OB_TEMP)
					{
						childNodes[2]->obIn = OB_TEMP;
						childNodes[2]->obOut = OB_USER_OUT;

						childNodes[3]->obIn = OB_USER_OUT;
						childNodes[3]->obOut = OB_USER_OUT;

						childNodes[4]->obIn = OB_USER_OUT;
						childNodes[4]->obOut = OB_TEMP;
					}
					else
					{
						childNodes[2]->obIn = OB_USER_OUT;
						childNodes[2]->obOut = OB_TEMP;

						childNodes[3]->obIn = OB_TEMP;
						childNodes[3]->obOut = OB_USER_OUT;

						childNodes[4]->obIn = OB_USER_OUT;
						childNodes[4]->obOut = OB_TEMP;
					}
				}
			}
		}
		else if(scheme == CS_L1D_CC)
		{
			if ((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
			{
				if (parent == nullptr)
				{
					childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
					childNodes[0]->obOut = OB_TEMP;

					childNodes[1]->obIn = OB_TEMP;
					childNodes[1]->obOut = OB_USER_OUT;
				}
				else
				{
					childNodes[0]->obIn = flipIn;
					childNodes[0]->obOut = flipOut;

					childNodes[1]->obIn = flipOut;
					childNodes[1]->obOut = flipIn;
				}

				obIn = childNodes[0]->obIn;
				obOut = childNodes[1]->obOut;
			}
			else
			{
				assert(obIn == flipIn);
				assert(obIn == obOut);

				childNodes[0]->obIn = flipIn;
				childNodes[0]->obOut = flipOut;

				childNodes[1]->obIn = flipOut;
				childNodes[1]->obOut = flipIn;
			}
				
		}
		else if (scheme == CS_L1D_CRT)
		{
			if ((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
			{
				if (parent == nullptr)
				{
					childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
					childNodes[0]->obOut = OB_TEMP;

					childNodes[1]->obIn = OB_TEMP;
					childNodes[1]->obOut = OB_TEMP;

					childNodes[2]->obIn = OB_TEMP;
					childNodes[2]->obOut = OB_USER_OUT;
				}
				else
				{
					childNodes[0]->obIn = flipIn;
					childNodes[0]->obOut = flipOut;

					childNodes[1]->obIn = flipOut;
					childNodes[1]->obOut = flipOut;

					childNodes[2]->obIn = flipOut;
					childNodes[2]->obOut = flipIn;
				}

				obIn = childNodes[0]->obIn;
				obOut = childNodes[2]->obOut;
			}
			else
			{
				assert(obIn == flipIn);
				assert(obIn == obOut);

				childNodes[0]->obIn = flipIn;
				childNodes[0]->obOut = flipOut;

				childNodes[1]->obIn = flipOut;
				childNodes[1]->obOut = flipOut;

				childNodes[2]->obIn = flipOut;
				childNodes[2]->obOut = flipIn;
			}
		}
		else if ( (scheme == CS_2D_RTRT) || (scheme == CS_3D_RTRT) )
		{
			if (parent == nullptr)
				childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
			else
				childNodes[0]->obIn = OB_USER_OUT;

			childNodes[0]->obOut = OB_USER_OUT;

			flipIn = OB_USER_OUT;
			flipOut = OB_TEMP;
			childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);

			childNodes[1]->obIn = OB_USER_OUT;
			childNodes[1]->obOut = OB_TEMP;

			childNodes[2]->obIn = OB_TEMP;
			childNodes[2]->obOut = OB_TEMP;

			flipIn = OB_TEMP;
			flipOut = OB_USER_OUT;
			childNodes[2]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);

			childNodes[3]->obIn = OB_TEMP;
			childNodes[3]->obOut = OB_USER_OUT;

			obIn = childNodes[0]->obIn;
			obOut = childNodes[3]->obOut;
		}
		else if ( (scheme == CS_2D_RC) || (scheme == CS_3D_RC) )
		{
			if (parent == nullptr)
				childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
			else
				childNodes[0]->obIn = OB_USER_OUT;

			childNodes[0]->obOut = OB_USER_OUT;

			flipIn = OB_USER_OUT;
			flipOut = OB_TEMP;
			childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);

			childNodes[1]->obIn = OB_USER_OUT;
			childNodes[1]->obOut = OB_USER_OUT;

			obIn = childNodes[0]->obIn;
			obOut = childNodes[1]->obOut;
		}
		else
		{
			if (parent == nullptr)
			{
				obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
				obOut = OB_USER_OUT;
			}
			else
			{
				if( (obIn == OB_UNINIT) || (obOut == OB_UNINIT) )
					assert(false);

				if (obIn != obOut)
				{
					OperatingBuffer t;
					t = flipIn;
					flipIn = flipOut;
					flipOut = t;
				}
			}
		}
	}
	
	void TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn, rocfft_array_type rootOut)
	{
		if (parent != nullptr)
		{
			placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;

			switch (obIn)
			{
			case OB_USER_IN: inArrayType = rootIn; break;
			case OB_USER_OUT: inArrayType = rootOut; break;
			case OB_TEMP: inArrayType = rocfft_array_type_complex_interleaved; break;
			default: inArrayType = rocfft_array_type_complex_interleaved;
			}

			switch (obOut)
			{
			case OB_USER_IN: assert(false); break;
			case OB_USER_OUT: outArrayType = rootOut; break;
			case OB_TEMP: outArrayType = rocfft_array_type_complex_interleaved; break;
			default: outArrayType = rocfft_array_type_complex_interleaved;
			}
		}

		std::vector<TreeNode *>::iterator children_p;
		for (children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
		{
			(*children_p)->TraverseTreeAssignPlacementsLogicA(rootIn, rootOut);
		}
	}

	void TraverseTreeAssignParamsLogicA()
	{

		switch (scheme)
		{
		case CS_L1D_TRTRT:
		{
			size_t biggerDim = childNodes[0]->length[0] > childNodes[0]->length[1] ? childNodes[0]->length[0] : childNodes[0]->length[1];
			size_t smallerDim = biggerDim == childNodes[0]->length[0] ? childNodes[0]->length[1] : childNodes[0]->length[0];
			size_t padding = 0;
			if ( ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512) )
				padding = 64;

			TreeNode *trans1Plan = childNodes[0];
			TreeNode *row1Plan = childNodes[1];
			TreeNode *trans2Plan = childNodes[2];
			TreeNode *row2Plan = childNodes[3];
			TreeNode *trans3Plan = childNodes[4];

			trans1Plan->inStride.push_back(inStride[0]);
			trans1Plan->inStride.push_back(trans1Plan->length[0]);
			trans1Plan->iDist = iDist;
			for (size_t index = 1; index < length.size(); index++) trans1Plan->inStride.push_back(inStride[index]);

			if (trans1Plan->obOut == OB_TEMP)
			{
				trans1Plan->outStride.push_back(1);
				trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
				trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];

				for (size_t index = 1; index < length.size(); index++)
				{
					trans1Plan->outStride.push_back(trans1Plan->oDist);
					trans1Plan->oDist *= length[index];
				}
			}
			else
			{
				if (parent->scheme == CS_L1D_TRTRT)
				{
					trans1Plan->outStride.push_back(outStride[0]);
					trans1Plan->outStride.push_back(outStride[0] * (trans1Plan->length[1]));
					trans1Plan->oDist = oDist;

					for (size_t index = 1; index < length.size(); index++) trans1Plan->outStride.push_back(outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					assert(parent->obOut == OB_USER_OUT);

					assert(parent->outStride[0] == 1);
					for (size_t index = 1; index < parent->length.size(); index++)
						assert(parent->outStride[index] == (parent->outStride[index - 1] * parent->length[index - 1]));


					trans1Plan->outStride.push_back(1);
					trans1Plan->outStride.push_back(trans1Plan->length[1]);
					trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->length[1];

					for (size_t index = 1; index < length.size(); index++)
					{
						trans1Plan->outStride.push_back(trans1Plan->oDist);
						trans1Plan->oDist *= length[index];
					}
				}
			}

			row1Plan->inStride = trans1Plan->outStride;
			row1Plan->iDist = trans1Plan->oDist;

			if (row1Plan->placement == rocfft_placement_inplace)
			{
				row1Plan->outStride = row1Plan->inStride;
				row1Plan->oDist = row1Plan->iDist;
			}
			else
			{
				assert(row1Plan->obOut == OB_USER_OUT);

				row1Plan->outStride.push_back(outStride[0]);
				row1Plan->outStride.push_back(outStride[0] * row1Plan->length[0]);
				row1Plan->oDist = oDist;

				for (size_t index = 1; index < length.size(); index++) row1Plan->outStride.push_back(outStride[index]);
			}

			row1Plan->TraverseTreeAssignParamsLogicA();

			trans2Plan->inStride = row1Plan->outStride;
			trans2Plan->iDist = row1Plan->oDist;

			if (trans2Plan->obOut == OB_TEMP)
			{
				trans2Plan->outStride.push_back(1);
				trans2Plan->outStride.push_back(trans2Plan->length[1] + padding);
				trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->outStride[1];

				for (size_t index = 1; index < length.size(); index++)
				{
					trans2Plan->outStride.push_back(trans2Plan->oDist);
					trans2Plan->oDist *= length[index];
				}
			}
			else
			{
				if ( (parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)) )
				{
					trans2Plan->outStride.push_back(outStride[0]);
					trans2Plan->outStride.push_back(outStride[0] * (trans2Plan->length[1]));
					trans2Plan->oDist = oDist;

					for (size_t index = 1; index < length.size(); index++) trans2Plan->outStride.push_back(outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					trans2Plan->outStride.push_back(1);
					trans2Plan->outStride.push_back(trans2Plan->length[1]);
					trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->length[1];

					for (size_t index = 1; index < length.size(); index++)
					{
						trans2Plan->outStride.push_back(trans2Plan->oDist);
						trans2Plan->oDist *= length[index];
					}
				}
			}

			row2Plan->inStride = trans2Plan->outStride;
			row2Plan->iDist = trans2Plan->oDist;

			if (row2Plan->obIn == row2Plan->obOut)
			{
				row2Plan->outStride = row2Plan->inStride;
				row2Plan->oDist = row2Plan->iDist;
			}
			else if (row2Plan->obOut == OB_TEMP)
			{
				row2Plan->outStride.push_back(1);
				row2Plan->outStride.push_back(row2Plan->length[0] + padding);
				row2Plan->oDist = row2Plan->length[1] * row2Plan->outStride[1];

				for (size_t index = 1; index < length.size(); index++)
				{
					row2Plan->outStride.push_back(row2Plan->oDist);
					row2Plan->oDist *= length[index];
				}
			}
			else
			{
				if ((parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)))
				{
					row2Plan->outStride.push_back(outStride[0]);
					row2Plan->outStride.push_back(outStride[0] * (row2Plan->length[0]));
					row2Plan->oDist = oDist;

					for (size_t index = 1; index < length.size(); index++) row2Plan->outStride.push_back(outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					row2Plan->outStride.push_back(1);
					row2Plan->outStride.push_back(row2Plan->length[0]);
					row2Plan->oDist = row2Plan->length[0] * row2Plan->length[1];

					for (size_t index = 1; index < length.size(); index++)
					{
						row2Plan->outStride.push_back(row2Plan->oDist);
						row2Plan->oDist *= length[index];
					}
				}
			}


			trans3Plan->inStride = row2Plan->outStride;
			trans3Plan->iDist = row2Plan->oDist;
			
			trans3Plan->outStride.push_back(outStride[0]);
			trans3Plan->outStride.push_back(outStride[0] * (trans3Plan->length[1]));
			trans3Plan->oDist = oDist;

			for (size_t index = 1; index < length.size(); index++) trans3Plan->outStride.push_back(outStride[index]);

		}
		break;
		case CS_L1D_CC:
		{
			TreeNode *col2colPlan = childNodes[0];
			TreeNode *row2colPlan = childNodes[1];

			if(parent != NULL) assert(obIn == obOut);

			if (obOut == OB_USER_OUT)
			{
				// B -> T
				col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
				col2colPlan->inStride.push_back(inStride[0]);
				col2colPlan->iDist = iDist;

				col2colPlan->outStride.push_back(col2colPlan->length[1]);
				col2colPlan->outStride.push_back(1);
				col2colPlan->oDist = length[0];

				for (size_t index = 1; index < length.size(); index++)
				{
					col2colPlan->inStride.push_back(inStride[index]);
					col2colPlan->outStride.push_back(col2colPlan->oDist);
					col2colPlan->oDist *= length[index];
				}

				// T -> B
				row2colPlan->inStride.push_back(1);
				row2colPlan->inStride.push_back(row2colPlan->length[0]);
				row2colPlan->iDist = length[0];

				row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
				row2colPlan->outStride.push_back(outStride[0]);
				row2colPlan->oDist = oDist;

				for (size_t index = 1; index < length.size(); index++)
				{
					row2colPlan->inStride.push_back(row2colPlan->iDist);
					row2colPlan->iDist *= length[index];
					row2colPlan->outStride.push_back(outStride[index]);
				}
			}
			else
			{
				// here we don't have B info right away, we get it through its parent
				assert(parent->obOut == OB_USER_OUT);

				// T-> B
				col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
				col2colPlan->inStride.push_back(inStride[0]);
				col2colPlan->iDist = iDist;

				for (size_t index = 1; index < length.size(); index++) col2colPlan->inStride.push_back(inStride[index]);

				if (parent->scheme == CS_L1D_TRTRT)
				{
					col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
					col2colPlan->outStride.push_back(parent->outStride[0]);
					col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1] * col2colPlan->length[0]);
					col2colPlan->oDist = parent->oDist;

					for (size_t index = 1; index < parent->length.size(); index++) col2colPlan->outStride.push_back(parent->outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					assert(parent->outStride[0] == 1);
					for (size_t index = 1; index < parent->length.size(); index++)
						assert(parent->outStride[index] == (parent->outStride[index-1] * parent->length[index-1]));

					col2colPlan->outStride.push_back(col2colPlan->length[1]);
					col2colPlan->outStride.push_back(1);
					col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

					for (size_t index = 1; index < length.size(); index++)
					{
						col2colPlan->outStride.push_back(col2colPlan->oDist);
						col2colPlan->oDist *= length[index];
					}
				}

				// B -> T
				if (parent->scheme == CS_L1D_TRTRT)
				{
					row2colPlan->inStride.push_back(parent->outStride[0]);
					row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0]);
					row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0] * row2colPlan->length[1]);
					row2colPlan->iDist = parent->oDist;

					for (size_t index = 1; index < parent->length.size(); index++) row2colPlan->inStride.push_back(parent->outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					row2colPlan->inStride.push_back(1);
					row2colPlan->inStride.push_back(row2colPlan->length[0]);
					row2colPlan->iDist = row2colPlan->length[0] * row2colPlan->length[1];
					
					for (size_t index = 1; index < length.size(); index++)
					{
						row2colPlan->inStride.push_back(row2colPlan->iDist);
						row2colPlan->iDist *= length[index];
					}
				}

				row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
				row2colPlan->outStride.push_back(outStride[0]);
				row2colPlan->oDist = oDist;

				for (size_t index = 1; index < length.size(); index++) row2colPlan->outStride.push_back(outStride[index]);
			}
		}
		break;
		case CS_L1D_CRT:
		{
			TreeNode *col2colPlan = childNodes[0];
			TreeNode *row2rowPlan = childNodes[1];
			TreeNode *transPlan = childNodes[2];

			if (parent != NULL) assert(obIn == obOut);

			if (obOut == OB_USER_OUT)
			{
				// B -> T
				col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
				col2colPlan->inStride.push_back(inStride[0]);
				col2colPlan->iDist = iDist;

				col2colPlan->outStride.push_back(col2colPlan->length[1]);
				col2colPlan->outStride.push_back(1);
				col2colPlan->oDist = length[0];

				for (size_t index = 1; index < length.size(); index++)
				{
					col2colPlan->inStride.push_back(inStride[index]);
					col2colPlan->outStride.push_back(col2colPlan->oDist);
					col2colPlan->oDist *= length[index];
				}

				// T -> T
				row2rowPlan->inStride.push_back(1);
				row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
				row2rowPlan->iDist = length[0];

				for (size_t index = 1; index < length.size(); index++)
				{
					row2rowPlan->inStride.push_back(row2rowPlan->iDist);
					row2rowPlan->iDist *= length[index];
				}

				row2rowPlan->outStride = row2rowPlan->inStride;
				row2rowPlan->oDist = row2rowPlan->iDist;

				// T -> B
				transPlan->inStride = row2rowPlan->outStride;
				transPlan->iDist = row2rowPlan->oDist;

				transPlan->outStride.push_back(outStride[0]);
				transPlan->outStride.push_back(outStride[0] * (transPlan->length[1]));
				transPlan->oDist = oDist;

				for (size_t index = 1; index < length.size(); index++) transPlan->outStride.push_back(outStride[index]);
			}
			else
			{
				// here we don't have B info right away, we get it through its parent
				assert(parent->obOut == OB_USER_OUT);

				// T -> B
				col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
				col2colPlan->inStride.push_back(inStride[0]);
				col2colPlan->iDist = iDist;

				for (size_t index = 1; index < length.size(); index++) col2colPlan->inStride.push_back(inStride[index]);

				if (parent->scheme == CS_L1D_TRTRT)
				{
					col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
					col2colPlan->outStride.push_back(parent->outStride[0]);
					col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1] * col2colPlan->length[0]);
					col2colPlan->oDist = parent->oDist;

					for (size_t index = 1; index < parent->length.size(); index++) col2colPlan->outStride.push_back(parent->outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					assert(parent->outStride[0] == 1);
					for (size_t index = 1; index < parent->length.size(); index++)
						assert(parent->outStride[index] == (parent->outStride[index - 1] * parent->length[index - 1]));

					col2colPlan->outStride.push_back(col2colPlan->length[1]);
					col2colPlan->outStride.push_back(1);
					col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

					for (size_t index = 1; index < length.size(); index++)
					{
						col2colPlan->outStride.push_back(col2colPlan->oDist);
						col2colPlan->oDist *= length[index];
					}
				}

				// B -> B
				if (parent->scheme == CS_L1D_TRTRT)
				{
					row2rowPlan->inStride.push_back(parent->outStride[0]);
					row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0]);
					row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0] * row2rowPlan->length[1]);
					row2rowPlan->iDist = parent->oDist;

					for (size_t index = 1; index < parent->length.size(); index++) row2rowPlan->inStride.push_back(parent->outStride[index]);
				}
				else
				{
					// we dont have B info here, need to assume packed data and descended from 2D/3D
					row2rowPlan->inStride.push_back(1);
					row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
					row2rowPlan->iDist = row2rowPlan->length[0] * row2rowPlan->length[1];

					for (size_t index = 1; index < length.size(); index++)
					{
						row2rowPlan->inStride.push_back(row2rowPlan->iDist);
						row2rowPlan->iDist *= length[index];
					}
				}

				row2rowPlan->outStride = row2rowPlan->inStride;
				row2rowPlan->oDist = row2rowPlan->iDist;

				// B -> T
				transPlan->inStride = row2rowPlan->outStride;
				transPlan->iDist = row2rowPlan->oDist;

				transPlan->outStride.push_back(outStride[0]);
				transPlan->outStride.push_back(outStride[0] * transPlan->length[1]);
				transPlan->oDist = oDist;

				for (size_t index = 1; index < length.size(); index++) transPlan->outStride.push_back(outStride[index]);
			}
		}
		break;
		case CS_2D_RTRT:
		{
			TreeNode *row1Plan = childNodes[0];
			TreeNode *trans1Plan = childNodes[1];
			TreeNode *row2Plan = childNodes[2];
			TreeNode *trans2Plan = childNodes[3];


			size_t biggerDim = length[0] > length[1] ? length[0] : length[1];
			size_t smallerDim = biggerDim == length[0] ? length[1] : length[0];
			size_t padding = 0;
			if (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512))
				padding = 64;

			// B -> B
			assert(row1Plan->obOut == OB_USER_OUT);			
			row1Plan->inStride = inStride;
			row1Plan->iDist = iDist;

			row1Plan->outStride = outStride;
			row1Plan->oDist = oDist;

			row1Plan->TraverseTreeAssignParamsLogicA();

			// B -> T
			assert(trans1Plan->obOut == OB_TEMP);
			trans1Plan->inStride = row1Plan->outStride;
			trans1Plan->iDist = row1Plan->oDist;

			trans1Plan->outStride.push_back(1);
			trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
			trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];

			for (size_t index = 2; index < length.size(); index++)
			{
				trans1Plan->outStride.push_back(trans1Plan->oDist);
				trans1Plan->oDist *= length[index];
			}

			// T -> T
			assert(row2Plan->obOut == OB_TEMP);
			row2Plan->inStride = trans1Plan->outStride;
			row2Plan->iDist = trans1Plan->oDist;

			row2Plan->outStride = row2Plan->inStride;
			row2Plan->oDist = row2Plan->iDist;

			row2Plan->TraverseTreeAssignParamsLogicA();

			// T -> B
			assert(trans2Plan->obOut == OB_USER_OUT);
			trans2Plan->inStride = row2Plan->outStride;
			trans2Plan->iDist = row2Plan->oDist;

			trans2Plan->outStride = outStride;
			trans2Plan->oDist = oDist;
		}
		break;
		case CS_2D_RC:
		case CS_2D_STRAIGHT:
		{
			TreeNode *rowPlan = childNodes[0];
			TreeNode *colPlan = childNodes[1];

			// B -> B
			assert(rowPlan->obOut == OB_USER_OUT);
			rowPlan->inStride = inStride;
			rowPlan->iDist = iDist;

			rowPlan->outStride = outStride;
			rowPlan->oDist = oDist;

			rowPlan->TraverseTreeAssignParamsLogicA();

			// B -> B
			assert(colPlan->obOut == OB_USER_OUT);
			colPlan->inStride.push_back(inStride[1]);
			colPlan->inStride.push_back(inStride[0]);
			for (size_t index = 2; index < length.size(); index++) colPlan->inStride.push_back(inStride[index]);

			colPlan->iDist = rowPlan->oDist;

			colPlan->outStride = colPlan->inStride;
			colPlan->oDist = colPlan->iDist;
		};
		break;
		case CS_3D_RTRT:
		{
			TreeNode *xyPlan = childNodes[0];
			TreeNode *trans1Plan = childNodes[1];
			TreeNode *zPlan = childNodes[2];
			TreeNode *trans2Plan = childNodes[3];


			size_t biggerDim = (length[0] * length[1]) > length[2] ? (length[0] * length[1]) : length[2];
			size_t smallerDim = biggerDim == (length[0] * length[1]) ? length[2] : (length[0] * length[1]);
			size_t padding = 0;
			if (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512))
				padding = 64;

			// B -> B
			assert(xyPlan->obOut == OB_USER_OUT);
			xyPlan->inStride = inStride;
			xyPlan->iDist = iDist;

			xyPlan->outStride = outStride;
			xyPlan->oDist = oDist;

			xyPlan->TraverseTreeAssignParamsLogicA();

			// B -> T
			assert(trans1Plan->obOut == OB_TEMP);
			trans1Plan->inStride = xyPlan->outStride;
			trans1Plan->iDist = xyPlan->oDist;

			trans1Plan->outStride.push_back(1);
			trans1Plan->outStride.push_back(trans1Plan->length[2] + padding);
			trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[1]);
			trans1Plan->oDist = trans1Plan->length[1] * trans1Plan->outStride[2];

			for (size_t index = 3; index < length.size(); index++)
			{
				trans1Plan->outStride.push_back(trans1Plan->oDist);
				trans1Plan->oDist *= length[index];
			}

			// T -> T
			assert(zPlan->obOut == OB_TEMP);
			zPlan->inStride = trans1Plan->outStride;
			zPlan->iDist = trans1Plan->oDist;

			zPlan->outStride = zPlan->inStride;
			zPlan->oDist = zPlan->iDist;

			zPlan->TraverseTreeAssignParamsLogicA();

			// T -> B
			assert(trans2Plan->obOut == OB_USER_OUT);
			trans2Plan->inStride = zPlan->outStride;
			trans2Plan->iDist = zPlan->oDist;

			trans2Plan->outStride = outStride;
			trans2Plan->oDist = oDist;
		}
		break;
		case CS_3D_RC:
		case CS_3D_STRAIGHT:
		{
			TreeNode *xyPlan = childNodes[0];
			TreeNode *zPlan = childNodes[1];

			// B -> B
			assert(xyPlan->obOut == OB_USER_OUT);
			xyPlan->inStride = inStride;
			xyPlan->iDist = iDist;

			xyPlan->outStride = outStride;
			xyPlan->oDist = oDist;

			xyPlan->TraverseTreeAssignParamsLogicA();

			// B -> B
			assert(zPlan->obOut == OB_USER_OUT);
			zPlan->inStride.push_back(inStride[2]);
			zPlan->inStride.push_back(inStride[0]);
			zPlan->inStride.push_back(inStride[1]);
			for (size_t index = 3; index < length.size(); index++) zPlan->inStride.push_back(inStride[index]);

			zPlan->iDist = xyPlan->oDist;

			zPlan->outStride = zPlan->inStride;
			zPlan->oDist = zPlan->iDist;
		};
		break;
		default: return;

		}
	}
	
	void TraverseTreeCollectLeafsLogicA(std::vector<TreeNode *> &seq, size_t &workBufSize)
	{
		if (childNodes.size() == 0)
		{
			assert(length.size() == inStride.size());
			assert(length.size() == outStride.size());

			if (obOut == OB_TEMP) workBufSize = oDist > workBufSize ? oDist : workBufSize;
			seq.push_back(this);
		}
		else
		{
			std::vector<TreeNode *>::iterator children_p;
			for (children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
			{
				(*children_p)->TraverseTreeCollectLeafsLogicA(seq, workBufSize);
			}
		}
	}


	void Print(int indent = 0)
	{
		std::string indentStr;
		int i = indent;
		while (i--) indentStr += "    ";

		std::cout << std::endl << indentStr.c_str();
		std::cout << "dimension: " << dimension;
		std::cout << std::endl << indentStr.c_str();
		std::cout << "length: " << length[0];
		for (size_t i = 1; i < length.size(); i++)
			std::cout << ", " << length[i];

		std::cout << std::endl << indentStr.c_str() << "iStrides: ";
		for (size_t i = 0; i < inStride.size(); i++)
			std::cout << inStride[i] << ", ";
		std::cout << iDist;

		std::cout << std::endl << indentStr.c_str() << "oStrides: ";
		for (size_t i = 0; i < outStride.size(); i++)
			std::cout << outStride[i] << ", ";
		std::cout << oDist;

		std::cout << std::endl << indentStr.c_str() << "format: " << placement << " " << inArrayType << " " << outArrayType;
		std::cout << std::endl << indentStr.c_str() << "scheme: " << PrintScheme(scheme).c_str() << std::endl << indentStr.c_str();

		if (obIn == OB_USER_IN) std::cout << "A -> ";
		else if (obIn == OB_USER_OUT) std::cout << "B -> ";
		else if (obIn == OB_TEMP) std::cout << "T -> ";
		else std::cout << "ERR -> ";

		if (obOut == OB_USER_IN) std::cout << "A";
		else if (obOut == OB_USER_OUT) std::cout << "B";
		else if (obOut == OB_TEMP) std::cout << "T";
		else std::cout << "ERR";

		std::cout << std::endl;
	
		if(childNodes.size())
		{
			std::vector<TreeNode *>::iterator children_p;
			for (children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
			{
				(*children_p)->Print(indent+1);
			}
		}
	}

	// logic B - using in-place transposes, todo
	void RecursiveBuildTreeLogicB()
	{

	}


};

void ProcessNode(TreeNode *rootPlan)
{
	std::cout << "*******************************************************************************" << std::endl;

	assert(rootPlan->length.size() == rootPlan->dimension);

	if (rootPlan->placement == rocfft_placement_inplace)
		assert(rootPlan->inArrayType == rootPlan->outArrayType);

	rootPlan->RecursiveBuildTree();
	OperatingBuffer flipIn, flipOut;
	rootPlan->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);
	rootPlan->TraverseTreeAssignPlacementsLogicA(rootPlan->inArrayType, rootPlan->outArrayType);
	rootPlan->TraverseTreeAssignParamsLogicA();

	std::vector<TreeNode *> execSeq;
	size_t workBufSize = 0;
	rootPlan->TraverseTreeCollectLeafsLogicA(execSeq, workBufSize);

	size_t N = 1;
	for (size_t i = 0; i < rootPlan->length.size(); i++) N *= rootPlan->length[i];
	std::cout << "Work buffer size: " << workBufSize << std::endl;
	std::cout << "Work buffer ratio: " << (double)workBufSize/(double)N << std::endl;

	if (execSeq.size() > 1)
	{
		std::vector<TreeNode *>::iterator prev_p = execSeq.begin();
		std::vector<TreeNode *>::iterator curr_p = prev_p + 1;
		while (curr_p != execSeq.end())
		{
			if ((*curr_p)->placement == rocfft_placement_inplace)
			{
				for (size_t i = 0; i < ((*curr_p)->inStride.size()); i++)
				{
					if (((*curr_p)->inStride[i]) != ((*curr_p)->outStride[i]))
						std::cout << "error in stride assignments" << std::endl;
					if (((*curr_p)->iDist) != ((*curr_p)->oDist))
						std::cout << "error in dist assignments" << std::endl;
				}

			}

			if ((*prev_p)->obOut != (*curr_p)->obIn)
				std::cout << "error in buffer assignments" << std::endl;

			prev_p = curr_p;
			curr_p++;
		}
	}

	rootPlan->Print();

	std::cout << "===============================================================================" << std::endl << std::endl;
}

// bake plan
void BakePlan()
{
	// 1 D
#if 0
	for (size_t i = 1; i <= 45; i++)
	{
		size_t N = (size_t)1 << i;
		TreeNode *rootPlan = TreeNode::CreateNode();

		rootPlan->dimension = 1;
		rootPlan->batchsize = 1;
		rootPlan->length.push_back(N);

		rootPlan->inStride.push_back(1);
		rootPlan->outStride.push_back(1);
		rootPlan->iDist = rootPlan->oDist = N;

		rootPlan->placement = rocfft_placement_inplace;
		rootPlan->precision = rocfft_precision_single;

		rootPlan->inArrayType  = rocfft_array_type_complex_interleaved;
		rootPlan->outArrayType = rocfft_array_type_complex_interleaved;

		ProcessNode(rootPlan);

		TreeNode::DeleteNode(rootPlan);
	}
#endif

	// 2 D
#if 0
	for (size_t j = 1; j <= 22; j++)
	{
		size_t N1 = (size_t)1 << j;
		for (size_t i = 1; i <= 22; i++)
		{
			size_t N0 = (size_t)1 << i;
			TreeNode *rootPlan = TreeNode::CreateNode();

			rootPlan->dimension = 2;
			rootPlan->batchsize = 1;
			rootPlan->length.push_back(N0);
			rootPlan->length.push_back(N1);

			rootPlan->inStride.push_back(1);
			rootPlan->inStride.push_back(N0);
			rootPlan->outStride.push_back(1);
			rootPlan->outStride.push_back(N0);
			rootPlan->iDist = rootPlan->oDist = N0*N1;

			rootPlan->placement = rocfft_placement_inplace;
			rootPlan->precision = rocfft_precision_single;

			rootPlan->inArrayType = rocfft_array_type_complex_interleaved;
			rootPlan->outArrayType = rocfft_array_type_complex_interleaved;

			ProcessNode(rootPlan);

			TreeNode::DeleteNode(rootPlan);
		}
	}
#endif

	// 3 D
#if 0
	for (size_t k = 1; k <= 10; k++)
	{
		size_t N2 = (size_t)1 << k;
		for (size_t j = 1; j <= 20; j++)
		{
			size_t N1 = (size_t)1 << j;
			for (size_t i = 1; i <= 10; i++)
			{
				size_t N0 = (size_t)1 << i;
				TreeNode *rootPlan = TreeNode::CreateNode();

				rootPlan->dimension = 3;
				rootPlan->batchsize = 1;
				rootPlan->length.push_back(N0);
				rootPlan->length.push_back(N1);
				rootPlan->length.push_back(N2);

				rootPlan->inStride.push_back(1);
				rootPlan->inStride.push_back(N0);
				rootPlan->inStride.push_back(N0*N1);
				rootPlan->outStride.push_back(1);
				rootPlan->outStride.push_back(N0);
				rootPlan->outStride.push_back(N0*N1);
				rootPlan->iDist = rootPlan->oDist = N0*N1*N2;

				rootPlan->placement = rocfft_placement_inplace;
				rootPlan->precision = rocfft_precision_single;

				rootPlan->inArrayType = rocfft_array_type_complex_interleaved;
				rootPlan->outArrayType = rocfft_array_type_complex_interleaved;

				ProcessNode(rootPlan);

				TreeNode::DeleteNode(rootPlan);
			}
		}
	}
#endif

}


