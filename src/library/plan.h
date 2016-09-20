
#ifndef PLAN_H
#define PLAN_H

#include <cstring>

#ifndef nullptr
#define nullptr NULL
#endif

#define MIN(A,B) (((A) < (B)) ? (A) : (B))

struct rocfft_plan_description_t
{

	rocfft_array_type inArrayType, outArrayType;

	size_t inStrides[3];
	size_t outStrides[3];

	size_t inDist;
	size_t outDist;

	size_t inOffset[2];
	size_t outOffset[2];

	double scale;

	rocfft_plan_description_t()
	{
		inArrayType  = rocfft_array_type_complex_interleaved;
		outArrayType = rocfft_array_type_complex_interleaved;

		inStrides[0] = 0;
		inStrides[1] = 0;
		inStrides[2] = 0;

		outStrides[0] = 0;
		outStrides[1] = 0;
		outStrides[2] = 0;

		inDist = 0;
		outDist = 0;

		inOffset[0]  = 0;
		inOffset[1]  = 0;
		outOffset[0] = 0;
		outOffset[1] = 0;

		scale = 1.0;
	}
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


	bool operator<(const rocfft_plan_t &b) const
	{
		const rocfft_plan_t &a = *this;

		return (memcmp(&a, &b, sizeof(rocfft_plan_t)) < 0 ? true : false);
	}
};


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

void *twiddles_create(size_t N);
void twiddles_delete(void *twt);

class TreeNode
{
private:
	// disallow public creation
	TreeNode(TreeNode *p) : parent(p), scheme(CS_NONE), obIn(OB_UNINIT), obOut(OB_UNINIT), large1D(0), twiddles(nullptr), twiddles_large(nullptr)
	{}

public:
	size_t						batch;

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

	void		*twiddles;
	void		*twiddles_large;

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
		if(!node)
			return;

		std::vector<TreeNode *>::iterator children_p;
		for (children_p = node->childNodes.begin(); children_p != node->childNodes.end(); children_p++)
			DeleteNode(*children_p); // recursively delete allocated nodes

		if(node->twiddles)
		{
			twiddles_delete(node->twiddles);
			node->twiddles = nullptr;
		}
		
		if(node->twiddles_large)
		{
			twiddles_delete(node->twiddles_large);
			node->twiddles_large = nullptr;
		}

		delete node;
	}

	void RecursiveBuildTree();
	void TraverseTreeAssignBuffersLogicA(OperatingBuffer &flipIn, OperatingBuffer &flipOut);
	void TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn, rocfft_array_type rootOut);
	void TraverseTreeAssignParamsLogicA();
	void TraverseTreeCollectLeafsLogicA(std::vector<TreeNode *> &seq, size_t &workBufSize);
	void Print(int indent = 0);

	// logic B - using in-place transposes, todo
	void RecursiveBuildTreeLogicB();
};



#endif // PLAN_H


