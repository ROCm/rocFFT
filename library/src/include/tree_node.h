
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <cstring>
#include <vector>

#include "twiddles.h"

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

enum TransTileDir
{
    TTD_IP_HOR,
    TTD_IP_VER,
};



class TreeNode
{
private:
    // disallow public creation
    TreeNode(TreeNode *p) : parent(p), scheme(CS_NONE), obIn(OB_UNINIT), obOut(OB_UNINIT), large1D(0),
                transTileDir(TTD_IP_HOR), twiddles(nullptr), twiddles_large(nullptr)
    {
        if(p != nullptr)
        {
            precision = p->precision;
            batch = p->batch;
            direction = p->direction;
        }
    }

public:
    size_t                        batch;

    // transform dimension - note this can be different from data dimension
    size_t                        dimension;

    // length of the FFT in each dimension
    std::vector< size_t >        length;

    // stride of the FFT in each dimension
    std::vector< size_t >        inStride, outStride;

    // distance between consecutive batch members
    size_t                        iDist, oDist;

    int                        direction;
    rocfft_result_placement        placement;
    rocfft_precision            precision;
    rocfft_array_type            inArrayType, outArrayType;

    // extra twiddle multiplication for large 1D
    size_t                        large1D;

    TreeNode                    *parent;
    std::vector<TreeNode *>        childNodes;

    ComputeScheme                scheme;
    OperatingBuffer                obIn, obOut;

    TransTileDir    transTileDir;

    void        *twiddles;
    void        *twiddles_large;

public:

    TreeNode(const TreeNode &) = delete;            // disallow copy constructor
    TreeNode& operator=(const TreeNode&) = delete;    // disallow assignment operator

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


extern "C"
{
typedef void (*DevFnCall)(void *, void *);
}


struct GridParam
{
    unsigned int b_x, b_y, b_z;//in HIP, the data type of dimensions of work items, work groups is unsigned int
    unsigned int tpb_x, tpb_y, tpb_z;

    GridParam() : b_x(1), b_y(1), b_z(1), tpb_x(1), tpb_y(1), tpb_z(1)
    {}
};

struct ExecPlan
{
    TreeNode *rootPlan;
    std::vector<TreeNode *> execSeq;
    std::vector<DevFnCall> devFnCall;
    std::vector<GridParam> gridParam;
    size_t workBufSize;

    ExecPlan() : rootPlan(nullptr), workBufSize(0)
    {}
};



void ProcessNode(ExecPlan &execPlan);
void PrintNode(ExecPlan &execPlan);


#endif // TREE_NODE_H
