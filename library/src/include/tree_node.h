// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "kargs.h"
#include "twiddles.h"

enum OperatingBuffer
{
    OB_UNINIT,
    OB_USER_IN,
    OB_USER_OUT,
    OB_TEMP,
    OB_TEMP_CMPLX_FOR_REAL,
    OB_TEMP_BLUESTEIN,
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

    CS_REAL_TRANSFORM_USING_CMPLX,
    CS_KERNEL_COPY_R_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_HERM,
    CS_KERNEL_COPY_HERM_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_R,

    CS_REAL_TRANSFORM_EVEN,
    CS_KERNEL_R_TO_CMPLX,
    CS_KERNEL_CMPLX_TO_R,
    CS_REAL_2D_EVEN,

    CS_BLUESTEIN,
    CS_KERNEL_CHIRP,
    CS_KERNEL_PAD_MUL,
    CS_KERNEL_FFT_MUL,
    CS_KERNEL_RES_MUL,

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
    TreeNode(TreeNode* p)
        : parent(p)
        , scheme(CS_NONE)
        , obIn(OB_UNINIT)
        , obOut(OB_UNINIT)
        , large1D(0)
        , lengthBlue(0)
        , iOffset(0)
        , oOffset(0)
        , iDist(0)
        , oDist(0)
        , transTileDir(TTD_IP_HOR)
        , twiddles(nullptr)
        , twiddles_large(nullptr)
        , devKernArg(nullptr)
        , inArrayType(rocfft_array_type_unset)
        , outArrayType(rocfft_array_type_unset)
    {
        if(p != nullptr)
        {
            precision = p->precision;
            batch     = p->batch;
            direction = p->direction;
        }

        Pow2Lengths1Single.insert(std::make_pair(8192, 64));
        Pow2Lengths1Single.insert(std::make_pair(16384, 64));
        Pow2Lengths1Single.insert(std::make_pair(32768, 128));
        Pow2Lengths1Single.insert(std::make_pair(65536, 256));
        Pow2Lengths1Single.insert(std::make_pair(131072, 64));
        Pow2Lengths1Single.insert(std::make_pair(262144, 64));

        Pow2Lengths1Double.insert(std::make_pair(4096, 64));
        Pow2Lengths1Double.insert(std::make_pair(8192, 64));
        Pow2Lengths1Double.insert(std::make_pair(16384, 64));
        Pow2Lengths1Double.insert(std::make_pair(32768, 128));
        Pow2Lengths1Double.insert(std::make_pair(65536, 64));
        Pow2Lengths1Double.insert(std::make_pair(131072, 64));
    }

    // Maps from length[0] to divLength1 for 1D transforms in
    // single and double precision for power-of-two transfor sizes
    // using blocks.
    std::map<size_t, size_t> Pow2Lengths1Single;
    std::map<size_t, size_t> Pow2Lengths1Double;

    // Compute divLength1 from Length[0] for non-power-of-two 1D
    // transform sizes
    size_t div1DNoPo2(const size_t length0);

public:
    size_t batch;

    // transform dimension - note this can be different from data dimension, user
    // provided
    size_t dimension;

    // length of the FFT in each dimension, internal value
    std::vector<size_t> length;

    // stride of the FFT in each dimension
    std::vector<size_t> inStride, outStride;

    // distance between consecutive batch members
    size_t iDist, oDist;

    int                     direction;
    rocfft_result_placement placement;
    rocfft_precision        precision;
    rocfft_array_type       inArrayType, outArrayType;

    // extra twiddle multiplication for large 1D
    size_t large1D;

    TreeNode*              parent;
    std::vector<TreeNode*> childNodes;

    ComputeScheme   scheme;
    OperatingBuffer obIn, obOut;

    TransTileDir transTileDir;

    size_t lengthBlue;
    size_t iOffset, oOffset;

    // these are device pointers
    void*   twiddles;
    void*   twiddles_large;
    size_t* devKernArg;

public:
    TreeNode(const TreeNode&) = delete; // disallow copy constructor
    TreeNode& operator=(const TreeNode&) = delete; // disallow assignment operator

    // create node (user level) using this function
    static TreeNode* CreateNode(TreeNode* parentNode = nullptr)
    {
        return new TreeNode(parentNode);
    }

    // destroy node by calling this function
    static void DeleteNode(TreeNode* node)
    {
        if(!node)
            return;

        std::vector<TreeNode*>::iterator children_p;
        for(children_p = node->childNodes.begin(); children_p != node->childNodes.end();
            children_p++)
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

        if(node->devKernArg)
        {
            kargs_delete(node->devKernArg);
            node->devKernArg = nullptr;
        }

        delete node;
    }

    // Main tree builder:
    void RecursiveBuildTree();

    // Real-complex and complex-real node builder:
    void build_real();
    void build_real_embed();
    void build_real_even_1D();
    void build_real_even_2D();

    // 1D node builders:
    void build_1D();
    void build_1DBluestein();
    void build_1DCS_L1D_TRTRT(const size_t divLength0, const size_t divLength1);
    void build_1DCS_L1D_CC(const size_t divLength0, const size_t divLength1);
    void build_1DCS_L1D_CRT(const size_t divLength0, const size_t divLength1);

    // 2D node builders:
    void build_CS_2D_RTRT();
    void build_CS_2D_RTRT_real();

    // Buffer assignment:
    void assign_buffers_CS_REAL_TRANSFORM_USING_CMPLX(OperatingBuffer& flipIn,
                                                      OperatingBuffer& flipOut,
                                                      OperatingBuffer& obOutBuf);
    void assign_buffers_CS_REAL_TRANSFORM_EVEN(OperatingBuffer& flipIn,
                                               OperatingBuffer& flipOut,
                                               OperatingBuffer& obOutBuf);
    void assign_buffers_CS_REAL_2D_EVEN(OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf);
    void assign_buffers_CS_BLUESTEIN(OperatingBuffer& flipIn,
                                     OperatingBuffer& flipOut,
                                     OperatingBuffer& obOutBuf);
    void assign_buffers_CS_L1D_TRTRT(OperatingBuffer& flipIn,
                                     OperatingBuffer& flipOut,
                                     OperatingBuffer& obOutBuf);
    void assign_buffers_CS_L1D_CC(OperatingBuffer& flipIn,
                                  OperatingBuffer& flipOut,
                                  OperatingBuffer& obOutBuf);
    void assign_buffers_CS_L1D_CRT(OperatingBuffer& flipIn,
                                   OperatingBuffer& flipOut,
                                   OperatingBuffer& obOutBuf);
    void assign_buffers_CS_RTRT(OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf);
    void assign_buffers_CS_RC(OperatingBuffer& flipIn,
                              OperatingBuffer& flipOut,
                              OperatingBuffer& obOutBuf);
    void TraverseTreeAssignBuffersLogicA(OperatingBuffer& flipIn,
                                         OperatingBuffer& flipOut,
                                         OperatingBuffer& obOutBuf);
    void TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn, rocfft_array_type rootOut);

    // Set strides and distances:
    void assign_params_CS_REAL_TRANSFORM_USING_CMPLX();
    void assign_params_CS_REAL_TRANSFORM_EVEN();
    void assign_params_CS_REAL_2D_EVEN();
    void assign_params_CS_L1D_CC();
    void assign_params_CS_L1D_CRT();
    void assign_params_CS_BLUESTEIN();
    void assign_params_CS_L1D_TRTRT();
    void assign_params_CS_2D_RTRT();
    void assign_params_CS_2D_RC_STRAIGHT();
    void assign_params_CS_3D_RTRT();
    void assign_params_CS_3D_RC_STRAIGHT();
    void TraverseTreeAssignParamsLogicA();

    // Determine work memory requirements:
    void TraverseTreeCollectLeafsLogicA(std::vector<TreeNode*>& seq,
                                        size_t&                 tmpBufSize,
                                        size_t&                 cmplxForRealSize,
                                        size_t&                 blueSize,
                                        size_t&                 chirpSize);

    // Output plan information for debug purposes:
    void Print(std::ostream& os = std::cout, int indent = 0) const;

    // logic B - using in-place transposes, todo
    //void RecursiveBuildTreeLogicB();
};

extern "C" {
typedef void (*DevFnCall)(const void*, void*);
}

struct GridParam
{
    unsigned int b_x, b_y, b_z; // in HIP, the data type of dimensions of work
    // items, work groups is unsigned int
    unsigned int tpb_x, tpb_y, tpb_z;

    GridParam()
        : b_x(1)
        , b_y(1)
        , b_z(1)
        , tpb_x(1)
        , tpb_y(1)
        , tpb_z(1)
    {
    }
};

struct ExecPlan
{
    TreeNode*              rootPlan;
    std::vector<TreeNode*> execSeq;
    std::vector<DevFnCall> devFnCall;
    std::vector<GridParam> gridParam;
    size_t                 workBufSize;
    size_t                 tmpWorkBufSize;
    size_t                 copyWorkBufSize;
    size_t                 blueWorkBufSize;
    size_t                 chirpWorkBufSize;

    ExecPlan()
        : rootPlan(nullptr)
        , workBufSize(0)
        , tmpWorkBufSize(0)
        , copyWorkBufSize(0)
        , blueWorkBufSize(0)
    {
    }
};

void ProcessNode(ExecPlan& execPlan);
void PrintNode(std::ostream& os, const ExecPlan& execPlan);

#endif // TREE_NODE_H
