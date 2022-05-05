// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#define GENERIC_BUF_ASSIGMENT 1

#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../../../shared/gpubuf.h"
#include "../device/kernels/callback.h"
#include "../device/kernels/common.h"
#include "kargs.h"
#include "rtc.h"
#include <hip/hip_runtime_api.h>

enum OperatingBuffer
{
    OB_UNINIT              = 0b00000,
    OB_USER_IN             = 0b00001,
    OB_USER_OUT            = 0b00010,
    OB_TEMP                = 0b00100,
    OB_TEMP_CMPLX_FOR_REAL = 0b01000,
    OB_TEMP_BLUESTEIN      = 0b10000,
};

enum ComputeScheme
{
    CS_NONE,
    CS_KERNEL_STOCKHAM,
    CS_KERNEL_STOCKHAM_BLOCK_CC,
    CS_KERNEL_STOCKHAM_BLOCK_RC,
    CS_KERNEL_STOCKHAM_BLOCK_CR,
    CS_KERNEL_TRANSPOSE,
    CS_KERNEL_TRANSPOSE_XY_Z,
    CS_KERNEL_TRANSPOSE_Z_XY,

    CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
    CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
    CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,

    CS_REAL_TRANSFORM_USING_CMPLX,
    CS_KERNEL_COPY_R_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_HERM,
    CS_KERNEL_COPY_HERM_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_R,

    CS_REAL_TRANSFORM_EVEN,
    CS_KERNEL_R_TO_CMPLX,
    CS_KERNEL_R_TO_CMPLX_TRANSPOSE,
    CS_KERNEL_CMPLX_TO_R,
    CS_KERNEL_TRANSPOSE_CMPLX_TO_R,
    CS_REAL_2D_EVEN,
    CS_REAL_3D_EVEN,
    CS_KERNEL_APPLY_CALLBACK,

    CS_BLUESTEIN,
    CS_KERNEL_CHIRP,
    CS_KERNEL_PAD_MUL,
    CS_KERNEL_FFT_MUL,
    CS_KERNEL_RES_MUL,

    CS_L1D_TRTRT,
    CS_L1D_CC,
    CS_L1D_CRT,

    CS_2D_STRAIGHT, // not implemented yet
    CS_2D_RTRT,
    CS_2D_RC,
    CS_KERNEL_2D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_2D_SINGLE,

    CS_3D_STRAIGHT, // not implemented yet
    CS_3D_TRTRTR,
    CS_3D_RTRT,
    CS_3D_BLOCK_RC,
    CS_3D_BLOCK_CR,
    CS_3D_RC,
    CS_KERNEL_3D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_3D_SINGLE // not implemented yet
};

enum NodeType
{
    NT_UNDEFINED, // un init
    NT_INTERNAL, // an internal node contains childrens
    NT_LEAF, // a leaf node represents a kernel and has no childrens
};

enum FuseType
{
    FT_TRANS_WITH_STOCKHAM, // T_R
    FT_STOCKHAM_WITH_TRANS, // R_T
    FT_STOCKHAM_WITH_TRANS_Z_XY, // R_T-Z_XY
    FT_STOCKHAM_WITH_TRANS_XY_Z, // R_T-XY_Z
    FT_R2C_TRANSPOSE, // post-r2c + transpose
    FT_TRANSPOSE_C2R, // transpose + pre-c2r
    FT_STOCKHAM_R2C_TRANSPOSE, // Stokham + post-r2c + transpose (Advance of FT_R2C_TRANSPOSE)
};

// TODO: move this to rocfft.h and allow users to select via plan description
// the decision strategy for buffer assigment
enum rocfft_optimize_strategy
{
    rocfft_optimize_min_buffer, // minimize number of buffers, possibly fewer fusions
    rocfft_optimize_balance, // balance between buffer and fusion
    rocfft_optimize_max_fusion, // maximize number of fusions, possibly more buffers
};

std::string PrintScheme(ComputeScheme cs);
std::string PrintOperatingBuffer(const OperatingBuffer ob);
std::string PrintOperatingBufferCode(const OperatingBuffer ob);
std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty);
std::string PrintDirectToFromRegMode(const DirectRegType ty);

typedef void (*DevFnCall)(const void*, void*);

struct GridParam
{
    unsigned int b_x, b_y, b_z; // in HIP, the data type of dimensions of work
    // items, work groups is unsigned int
    unsigned int wgs_x, wgs_y, wgs_z;
    unsigned int lds_bytes; // dynamic LDS allocation size

    GridParam()
        : b_x(1)
        , b_y(1)
        , b_z(1)
        , wgs_x(1)
        , wgs_y(1)
        , wgs_z(1)
        , lds_bytes(0)
    {
    }
};

static bool is_device_gcn_arch(const hipDeviceProp_t& prop, const std::string& cmpTarget)
{
    std::string archName(prop.gcnArchName);
    return archName.find(cmpTarget) != -1;
}

static bool is_diagonal_sbrc_3D_length(size_t len)
{
    // SBRC diagonal-transpose dimensions are currently 128, 256
    return len == 128 || len == 256;
}

static bool is_cube_size(const std::vector<size_t>& length)
{
    return length.size() == 3 && length[0] == length[1] && length[1] == length[2];
}

inline size_t sizeof_precision(rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return 2 * sizeof(float);
    case rocfft_precision_double:
        return 2 * sizeof(double);
    }
    assert(false);
    return 0;
}

class TreeNode;

// The mininal tree node data needed to decide the scheme
struct NodeMetaData
{
    size_t                  batch     = 1;
    size_t                  dimension = 1;
    std::vector<size_t>     length;
    std::vector<size_t>     outputLength;
    std::vector<size_t>     inStride, outStride;
    size_t                  iDist = 0, oDist = 0;
    size_t                  iOffset = 0, oOffset = 0;
    int                     direction    = -1;
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;
    hipDeviceProp_t         deviceProp   = {};
    bool                    rootIsC2C;

    explicit NodeMetaData(TreeNode* refNode);
};

class rocfft_ostream;

class FuseShim
{
    friend class NodeFactory;

protected:
    FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : fuseType(type)
        , nodes(components)
    {
        // default
        lastFusedNode = nodes.size() - 1;
    }

    // if these schemes can be fused
    virtual bool CheckSchemeFusable() = 0;

    bool schemeFusable = false;

public:
    FuseType fuseType;

    // nodes that contained in this shim
    std::vector<TreeNode*> nodes;

    // basically all fusion should be effectively-outofplace,
    // but TransC2R and R2CTrans can do some tricks
    bool   allowInplace   = true;
    size_t firstFusedNode = 0;
    size_t lastFusedNode;

public:
    // for the derived class
    virtual ~FuseShim() = default;

    // if the in/out buffer meets the placement requirement
    // the firstOBuffer is optional, used in R2CTrans only
    virtual bool
        PlacementFusable(OperatingBuffer iBuf, OperatingBuffer firstOBuf, OperatingBuffer lastOBuf);

    // return the result of CheckSchemeFusable
    bool IsSchemeFusable() const;

    // NB: Some fusions perform better or worse in different arch.
    //     We mark those exceptions from the execPlan.
    //     (We can only know the arch name from execPlan)
    //     A known case is RTFuse, length 168 in MI50
    void OverwriteFusableFlag(bool fusable);

    void ForEachNode(std::function<void(TreeNode*)> func);

    // the first/last node that to be fused
    // for R_T, T_R, it is pretty simple [0] and [1]
    // but for R_T-Z_XY, we store an extra "pre-node" to test if the RT fuse can be done
    // in this case, the first, last are [1], [2]. [0] doesn't participate the fusion
    virtual TreeNode* FirstFuseNode() const;
    virtual TreeNode* LastFuseNode() const;

    virtual std::unique_ptr<TreeNode> FuseKernels() = 0;
};

class TreeNode
{
    friend class NodeFactory;

protected:
    TreeNode(TreeNode* p)
        : parent(p)
    {
        if(p != nullptr)
        {
            precision  = p->precision;
            batch      = p->batch;
            direction  = p->direction;
            deviceProp = p->deviceProp;
        }

        allowedOutBuf
            = OB_USER_IN | OB_USER_OUT | OB_TEMP | OB_TEMP_CMPLX_FOR_REAL | OB_TEMP_BLUESTEIN;

        allowedOutArrayTypes = {rocfft_array_type_complex_interleaved,
                                rocfft_array_type_complex_planar,
                                rocfft_array_type_real,
                                rocfft_array_type_hermitian_interleaved};
    }

public:
    // node type: internal node or leaf node, or un-defined (un-init)
    NodeType nodeType = NT_UNDEFINED;

    // Batch size
    size_t batch = 1;

    // Transform dimension - note this can be different from data dimension, user
    // provided
    size_t dimension = 1;

    // Length of the FFT in each dimension, internal value
    std::vector<size_t> length;

    // Row-major output lengths, from fastest to slowest.  If empty,
    // output length is assumed to be the same as input length.
    //
    // This is set for nodes that might do non-obvious things with
    // strides (e.g. having fastest dimension not be first), so that
    // buffer assignment can know whether a node's output will fit in
    // a given buffer.
    std::vector<size_t> outputLength;

    // Stride of the FFT in each dimension
    std::vector<size_t> inStride, outStride;

    // Distance between consecutive batch members:
    size_t iDist = 0, oDist = 0;

    // Offsets to start of data in buffer:
    size_t iOffset = 0, oOffset = 0;

    // Direction of the transform (-1: forward, +1: inverse)
    int direction = -1;

    // The number of padding at the end of each row in lds
    unsigned int lds_padding = 0;

    // Data format parameters:
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;

    // Extra twiddle multiplication for large 1D
    size_t large1D = 0;
    // decompose large twiddle to product of 256(8) or 128(7) or 64(6)...or 16(4)
    // default is 8, and sbcc could be dynamically decomposed
    size_t largeTwdBase = 8;
    // flag indicating if using the 3-step decomp. for large twiddle? (16^3, 32^3, 64^3)
    // if false, always use 8 as the base (256*256*256....)
    bool largeTwd3Steps = false;
    // "Steps": how many exact loops we need to decompose the LTWD?
    // if we pass this as a template arg in kernel, should avoid dynamic while-loop
    // We will update this in set_large_twd_base_steps()
    size_t ltwdSteps = 0;

    // embedded C2R/R2C pre/post processing
    EmbeddedType ebtype = EmbeddedType::NONE;

    // if the kernel supports/use/not-use dir-to-from-reg
    DirectRegType dir2regMode = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;

    // sbrc transpose type
    SBRC_TRANSPOSE_TYPE sbrcTranstype = SBRC_TRANSPOSE_TYPE::NONE;

    // Tree structure:
    // non-owning pointer to parent node, may be null
    TreeNode* parent = nullptr;
    // owned pointers to children
    std::vector<std::unique_ptr<TreeNode>> childNodes;

    // one shim is a group of several "possibly" fusable nodes
    std::vector<std::unique_ptr<FuseShim>> fuseShims;

    // FIXME: document
    ComputeScheme   scheme = CS_NONE;
    OperatingBuffer obIn = OB_UNINIT, obOut = OB_UNINIT;

    // FIXME: document
    size_t lengthBlue = 0;

    // Device pointers:
    // twiddle memory is owned by the repo
    void*            twiddles            = nullptr;
    size_t           twiddles_size       = 0;
    void*            twiddles_large      = nullptr;
    size_t           twiddles_large_size = 0;
    gpubuf_t<size_t> devKernArg;

    // callback parameters
    UserCallbacks callbacks;

    hipDeviceProp_t deviceProp = {};

    // comments inserted by optimization passes to explain changes done
    // to the node
    std::vector<std::string> comments;

    // runtime-compiled kernels for this node
    std::shared_future<std::unique_ptr<RTCKernel>> compiledKernel;
    std::shared_future<std::unique_ptr<RTCKernel>> compiledKernelWithCallbacks;

    // Does this node allow inplace/not-inplace? default true,
    // each class handles the exception
    // transpose, sbrc, fused stockham only outofplace
    // bluestein component leaf node (multiply) only inplace
    bool allowInplace    = true;
    bool allowOutofplace = true;

    size_t                      allowedOutBuf;
    std::set<rocfft_array_type> allowedOutArrayTypes;

public:
    // Disallow copy constructor:
    TreeNode(const TreeNode&) = delete;

    // for the derived class
    virtual ~TreeNode();

    // Disallow assignment operator:
    TreeNode& operator=(const TreeNode&) = delete;

    // Copy data from another node (to a fused node)
    void CopyNodeData(const TreeNode& srcNode);

    // Copy data from the NodeMetaData (after deciding scheme)
    void CopyNodeData(const NodeMetaData& data);

    bool isPlacementAllowed(rocfft_result_placement) const;
    bool isOutBufAllowed(OperatingBuffer oB) const;
    bool isOutArrayTypeAllowed(rocfft_array_type) const;
    bool isRootNode() const;
    bool isLeafNode() const;

    // whether or not the input/output access pattern may benefit from padding
    virtual bool PaddingBenefitsInput()
    {
        return false;
    }
    virtual bool PaddingBenefitsOutput()
    {
        return false;
    }

    virtual void RecursiveBuildTree(); // Main tree builder: override by child
    virtual void SanityCheck();
    // If high dims are contiguous, we can collapse them to make offset
    // calculation simpler
    void CollapseContiguousDims();
    // Leaf nodes can override this to say what dims can be collapsed.
    // Return values are indexes into the length/stride arrays.
    virtual std::vector<size_t> CollapsibleDims()
    {
        return {};
    }

    // able to fuse CS_KERNEL_STOCKHAM and CS_KERNEL_TRANSPOSE_Z_XY ?
    bool fuse_CS_KERNEL_TRANSPOSE_Z_XY();
    // able to fuse CS_KERNEL_STOCKHAM and CS_KERNEL_TRANSPOSE_XY_Z ?
    bool fuse_CS_KERNEL_TRANSPOSE_XY_Z();
    // able to fuse STK, r2c, transp to CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY ?
    bool fuse_CS_KERNEL_STK_R2C_TRANSPOSE();

    void ApplyFusion();

#if !GENERIC_BUF_ASSIGMENT
    // State maintained while traversing the tree.
    //
    // Preparation and execution of the tree basically involves a
    // depth-first traversal.  At each step, the logic working on a
    // node could want to know details of:
    //
    // 1. the node itself (i.e. this)
    // 2. the node's parent (i.e. this->parent), if present
    // 3. the most recently traversed leaf node, which may be:
    //    - not present, or
    //    - an earlier sibling of this node, or
    //    - the last leaf visited from some other parent
    // 4. the root node's input/output parameters
    //
    // The TraverseState struct stores 3 and 4.
    struct TraverseState;
    // Assign the input buffer for this kernel
    void SetInputBuffer(TraverseState& state);

    // Buffer assignment:
    virtual void AssignBuffers(TraverseState&   state,
                               OperatingBuffer& flipIn,
                               OperatingBuffer& flipOut,
                               OperatingBuffer& obOutBuf);
    // Set placement variable and in/out array types
    virtual void TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn,
                                                    rocfft_array_type rootOut);
#endif
    void RefreshTree();

    // Set strides and distances:
    void AssignParams();

    // Collect LeadNodes and FuseShims:
    void CollectLeaves(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq);

    // Determine work memory requirements:
    void DetermineBufferMemory(size_t& tmpBufSize,
                               size_t& cmplxForRealSize,
                               size_t& blueSize,
                               size_t& chirpSize);

    // Output plan information for debug purposes:
    void Print(rocfft_ostream& os, int indent = 0) const;

    // logic B - using in-place transposes, todo
    //void RecursiveBuildTreeLogicB();

    void RecursiveRemoveNode(TreeNode* node);

    // insert a newNode before the node "pos"
    void RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode);

    TreeNode* GetPlanRoot();
    TreeNode* GetFirstLeaf();
    TreeNode* GetLastLeaf();
    TreeNode* GetBluesteinComponentParent();
    bool      IsLastLeafNodeOfBluesteinComponent();
    bool      IsRootPlanC2CTransform();

    // Set length of transpose kernel node, since those are easily
    // knowable just by looking at the scheme and they're used in
    // many plans.  Throws an exception if this is not a transpose
    // node.
    void SetTransposeOutputLength();

    // Get row-major output length of this node.
    std::vector<size_t> GetOutputLength() const
    {
        return outputLength.empty() ? length : outputLength;
    }
    // Padding needs matching stride + length to make its decisions.
    // For most nodes, outStride + length can be used together.  For
    // some nodes, outputLength is what matches outStride.
    virtual bool UseOutputLengthForPadding()
    {
        return false;
    }

    virtual bool KernelCheck()                                             = 0;
    virtual bool CreateDevKernelArgs()                                     = 0;
    virtual bool CreateTwiddleTableResource()                              = 0;
    virtual void SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp) = 0;

    // for 3D SBRC kernels, decide the transpose type based on the
    // block width and lengths that the block tiles need to align on.
    // default type is NONE, meaning this isn't a SBRC node
    virtual SBRC_TRANSPOSE_TYPE sbrc_transpose_type(unsigned int blockWidth) const
    {
        return NONE;
    }

    // Compute the large twd decomposition base
    void set_large_twd_base_steps(size_t largeTWDLength);

protected:
    virtual void BuildTree_internal() = 0;
#if !GENERIC_BUF_ASSIGMENT
    virtual void AssignBuffers_internal(TraverseState&   state,
                                        OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf)
        = 0;
#endif
    virtual void AssignParams_internal() = 0;
};

class InternalNode : public TreeNode
{
    friend class NodeFactory;

protected:
    explicit InternalNode(TreeNode* p)
        : TreeNode(p)
    {
        nodeType = NT_INTERNAL;
    }

    bool CreateDevKernelArgs() override
    {
        throw std::runtime_error("Shouldn't call CreateDevKernelArgs in a non-LeafNode");
        return false;
    }

    bool CreateTwiddleTableResource() override
    {
        throw std::runtime_error("Shouldn't call CreateTwiddleTableResource in a non-LeafNode");
        return false;
    }

    void SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp) override
    {
        throw std::runtime_error("Shouldn't call SetupGridParamAndFuncPtr in a non-LeafNode");
    }

public:
    bool KernelCheck() override
    {
        return true;
    }
};

class LeafNode : public InternalNode
{
    friend class NodeFactory;

protected:
    LeafNode(TreeNode* p, ComputeScheme s)
        : InternalNode(p)
    {
        nodeType = NT_LEAF;
        scheme   = s;
    }

    bool                externalKernel   = false;
    bool                need_twd_table   = false;
    bool                twd_no_radices   = false;
    bool                twd_attach_halfN = false;
    std::vector<size_t> kernelFactors    = {};
    size_t              bwd              = 1; // bwd, wgs, lds are for grid param lds_bytes
    size_t              wgs              = 0;
    size_t              lds              = 0;

    void BuildTree_internal() final {} // nothing to do in leaf node
#if !GENERIC_BUF_ASSIGMENT
    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
#endif
    void           AssignParams_internal() final {} // nothing to do in leaf node
    bool           CreateLargeTwdTable();
    virtual size_t GetTwiddleTableLength();
    // Limit length of generated twiddle table.  Default limit is 0,
    // which means to generate the full length of table.
    virtual size_t GetTwiddleTableLengthLimit()
    {
        return 0;
    }
    virtual void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) = 0;

public:
    bool         KernelCheck() override;
    void         SanityCheck() override;
    virtual bool CreateDevKernelArgs() override;
    bool         CreateTwiddleTableResource() override;
    void         SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp) override;
    void         GetKernelFactors();
};

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/
class TransposeNode : public LeafNode
{
    friend class NodeFactory;

protected:
    TransposeNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        allowInplace = false;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;

public:
    // Transpose tiles read more row-ish and write more column-ish.  So
    // assume output benefits more from padding than input.
    bool PaddingBenefitsOutput() override
    {
        // HACK: only assume we benefit if we have no large twiddle multiply.
        //
        // Since large twiddle multiply (i.e. middle T of L1D_TRTRT)
        // cannot be fused with an FFT kernel, we should not try too
        // hard to pad its output.  The other T nodes of that plan can
        // keep their buffer assigments so that padding doesn't upset
        // the current choice of which nodes we fuse.
        return large1D == 0;
    }
};

struct ExecPlan
{
    // shared pointer allows for ExecPlans to be copyable
    std::shared_ptr<TreeNode> rootPlan;

    // non-owning pointers to the leaf-node children of rootPlan, which
    // are the nodes that do actual work
    std::vector<TreeNode*> execSeq;

    // flattened potentially-fusable shims of rootPlan
    std::vector<FuseShim*> fuseShims;

    std::vector<DevFnCall> devFnCall;
    std::vector<GridParam> gridParam;

    hipDeviceProp_t deviceProp;

    std::vector<size_t> iLength;
    std::vector<size_t> oLength;

    // default: starting from ABT, balance buffers and fusions
    // we could allow users to set in the later PR
    rocfft_optimize_strategy assignOptStrategy = rocfft_optimize_balance;

    // these sizes count in complex elements
    size_t workBufSize      = 0;
    size_t tmpWorkBufSize   = 0;
    size_t copyWorkBufSize  = 0;
    size_t blueWorkBufSize  = 0;
    size_t chirpWorkBufSize = 0;

    size_t WorkBufBytes(size_t base_type_size) const
    {
        // base type is the size of one real, work buf counts in
        // complex numbers
        return workBufSize * 2 * base_type_size;
    }

    // for callbacks, work out which nodes of the plan are loading data
    // from global memory, and storing data to global memory
    std::pair<TreeNode*, TreeNode*> get_load_store_nodes() const;
};

void ProcessNode(ExecPlan& execPlan);
void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan);

#endif // TREE_NODE_H
