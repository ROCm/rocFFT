#include <vector>
#include <assert.h>
#include <iostream>

#include "rocfft.h"

#define nullptr NULL

size_t Large1DThreshold = 4096;

static inline bool IsPo2(size_t u) {
	return (u != 0) && (0 == (u & (u - 1)));
}

inline size_t PrecisionWidth(rocfft_precision_e pr)
{
	switch (pr)
	{
	case rocfft_precision_single:	return 1;
	case rocfft_precision_double:	return 2;
	default:		assert(false);	return 1;
	}
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
	CA_NONE,
	CA_KERNEL_STOCKHAM,
	CA_KERNEL_STOCKHAM_BLOCK,
	CA_KERNEL_TRANSPOSE,
	CA_L1D_TRTRT,
	CA_L1D_CC,
	CA_L1D_CRT,
};



class TreeNode
{
private:
	// disallow public creation
	TreeNode(TreeNode *p) : parent(p), scheme(CA_NONE), obIn(OB_UNINIT), obOut(OB_UNINIT)
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

	rocfft_result_placement_e	placement;
	rocfft_precision_e			precision;
	rocfft_array_type_e			inArrayType, outArrayType;

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

	// logic A - using out-of-place transposes & complex-to-complex & with padding
	void RecursiveBuildTreeLogicA()
	{
		switch (dimension)
		{
		case 1:
		{
			if (length[0] <= Large1DThreshold)
			{
				scheme = CA_KERNEL_STOCKHAM;
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

					scheme = (length[0] <= 65536 / PrecisionWidth(precision)) ? CA_L1D_CC : CA_L1D_CRT;
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

					scheme = CA_L1D_TRTRT;
				}
			}
			else
			{
			}

			size_t divLength0 = length[0] / divLength1;

			switch (scheme)
			{
			case CA_L1D_TRTRT:
			{
				// first transpose
				TreeNode *trans1Plan = TreeNode::CreateNode(this);
				trans1Plan->precision = precision;
				trans1Plan->batchsize = batchsize;

				trans1Plan->length.push_back(divLength0);
				trans1Plan->length.push_back(divLength1);

				trans1Plan->scheme = CA_KERNEL_TRANSPOSE;
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

				row1Plan->scheme = CA_KERNEL_STOCKHAM;
				row1Plan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row1Plan->length.push_back(length[index]);
				}

				row1Plan->RecursiveBuildTreeLogicA();
				childNodes.push_back(row1Plan);

				// second transpose
				TreeNode *trans2Plan = TreeNode::CreateNode(this);
				trans2Plan->precision = precision;
				trans2Plan->batchsize = batchsize;

				trans2Plan->length.push_back(divLength1);
				trans2Plan->length.push_back(divLength0);

				trans2Plan->scheme = CA_KERNEL_TRANSPOSE;
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

				row2Plan->scheme = CA_KERNEL_STOCKHAM;
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

				trans3Plan->scheme = CA_KERNEL_TRANSPOSE;
				trans3Plan->dimension = 2;

				for (size_t index = 1; index < length.size(); index++)
				{
					trans3Plan->length.push_back(length[index]);
				}			
				
				childNodes.push_back(trans3Plan);
			}
			break;
			case CA_L1D_CC:
			{
				// first plan, column-to-column
				TreeNode *col2colPlan = TreeNode::CreateNode(this);
				col2colPlan->precision = precision;
				col2colPlan->batchsize = batchsize;

				// large1D flag to confirm we need multiply twiddle factor
				col2colPlan->large1D = length[0];

				col2colPlan->length.push_back(divLength1);
				col2colPlan->length.push_back(divLength0);

				col2colPlan->scheme = CA_KERNEL_STOCKHAM_BLOCK;
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

				row2colPlan->scheme = CA_KERNEL_STOCKHAM_BLOCK;
				row2colPlan->dimension = 1;

				for (size_t index = 1; index < length.size(); index++)
				{
					row2colPlan->length.push_back(length[index]);
				}

				childNodes.push_back(row2colPlan);
			}
			break;
			case CA_L1D_CRT:
			{
				// first plan, column-to-column
				TreeNode *col2colPlan = TreeNode::CreateNode(this);
				col2colPlan->precision = precision;
				col2colPlan->batchsize = batchsize;

				// large1D flag to confirm we need multiply twiddle factor
				col2colPlan->large1D = length[0];

				col2colPlan->length.push_back(divLength1);
				col2colPlan->length.push_back(divLength0);

				col2colPlan->scheme = CA_KERNEL_STOCKHAM_BLOCK;
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

				row2rowPlan->scheme = CA_KERNEL_STOCKHAM_BLOCK;
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

				transPlan->scheme = CA_KERNEL_TRANSPOSE;
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
			if (scheme == CA_KERNEL_TRANSPOSE)
				return;

		}		
		break;

		case 3:
		{
		}
		break;

		default: assert(false);
		}

	}

	void TraverseTreeAssignBuffersLogicA(OperatingBuffer &flipIn, OperatingBuffer &flipOut)
	{
		if (parent == nullptr)
		{
			flipIn = OB_USER_OUT;
			flipOut = OB_TEMP;
		}

		if (scheme == CA_L1D_TRTRT)
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
		else if(scheme == CA_L1D_CC)
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
		else if (scheme == CA_L1D_CRT)
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
			if (parent == nullptr)
			{
				obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
				obOut = OB_USER_OUT;
			}
			else
			{
				obIn = flipIn;
				obOut = flipOut;

				OperatingBuffer t;
				t = flipIn;
				flipIn = flipOut;
				flipOut = t;
			}

		}
	}
	
	void TraverseTreeCollectLeafsLogicA(std::vector<TreeNode *> &seq)
	{
		if (childNodes.size() == 0)
		{
			seq.push_back(this);
		}
		else
		{
			std::vector<TreeNode *>::iterator children_p;
			for (children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
			{
				(*children_p)->TraverseTreeCollectLeafsLogicA(seq);
			}
		}
	}


	void Print(int indent = 0)
	{
		std::string indentStr;
		int i = indent;
		while (i--) indentStr += "    ";

		std::cout << std::endl << indentStr.c_str();
		std::cout << "length: ";
		for (size_t i = 0; i < length.size(); i++)
			std::cout << length[i] << ", ";
		std::cout << std::endl << indentStr.c_str() << "scheme: " << scheme << std::endl << indentStr.c_str();
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

// bake plan
void BakePlan()
{
	for (size_t i = 1; i <= 45; i++)
	{
		size_t N = (size_t)1 << i;
		TreeNode *rootPlan = TreeNode::CreateNode();

		rootPlan->dimension = 1;
		rootPlan->length.push_back(N);

		rootPlan->inStride.push_back(1);
		rootPlan->outStride.push_back(1);
		rootPlan->iDist = rootPlan->oDist = N;

		rootPlan->placement = rocfft_placement_inplace;
		rootPlan->precision = rocfft_precision_single;

		rootPlan->RecursiveBuildTreeLogicA();

		OperatingBuffer flipIn, flipOut;
		rootPlan->TraverseTreeAssignBuffersLogicA(flipIn, flipOut);

		std::vector<TreeNode *> execSeq;
		rootPlan->TraverseTreeCollectLeafsLogicA(execSeq);

		if (execSeq.size() > 1)
		{
			std::vector<TreeNode *>::iterator prev_p = execSeq.begin();
			std::vector<TreeNode *>::iterator curr_p = prev_p + 1;
			while (curr_p != execSeq.end())
			{
				if ((*prev_p)->obOut != (*curr_p)->obIn)
					std::cout << "error in buffer assignments" << std::endl;

				prev_p = curr_p;
				curr_p++;
			}
		}

		rootPlan->Print();

		TreeNode::DeleteNode(rootPlan);
	}

}

