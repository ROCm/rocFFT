// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "assignment_policy.h"
#include "./device/kernels/array_format.h"
#include "logging.h"
#include <set>

void PlacementTrace::Print(rocfft_ostream& os)
{
    if(parent->curNode)
    {
        parent->Print(os);
        os << " --> ";
    }

    os << "[ " << PrintScheme(curNode->scheme).c_str();
    os << ": " << PrintOperatingBufferCode(inBuf) << "->" << PrintOperatingBufferCode(outBuf);
    os << " ]";

    if(branches.empty())
    {
        os << ": num-fused-kernels= " << parent->numFusedNodes;
        os << ", num-inplace-kernels= " << parent->numInplace;
        os << std::endl;
    }
}

size_t PlacementTrace::BackwardCalcFusions(ExecPlan&       execPlan,
                                           int             curFuseShimID,
                                           PlacementTrace* shimLastNode)
{
    numFusedNodes = 0;
    if(curFuseShimID < 0)
        return 0;

    auto& shim = execPlan.fuseShims[curFuseShimID];

    // if this node is the last node of the fuseShim, pass self as shimLastNode and continue going back...
    if(curNode == shim->LastFuseNode())
    {
        // should not have shimLastNode set, and should not have a null parent
        if(shimLastNode != nullptr || parent == nullptr)
            throw std::runtime_error(
                "Tracing FusedShimsNode error when backtracking assignment path");
        numFusedNodes = parent->BackwardCalcFusions(execPlan, curFuseShimID, this);
    }
    // if this node is the first node of the fuseShim, check if fusion can be done with the placement
    else if(curNode == shim->FirstFuseNode())
    {
        // we should already have a shimLastNode
        if(!shimLastNode)
            throw std::runtime_error(
                "Tracing FusedShimsNode error when backtracking assignment path");
        size_t numFusion
            = shim->PlacementFusable(this->inBuf, this->outBuf, shimLastNode->outBuf) ? 1 : 0;
        numFusedNodes
            = parent ? parent->BackwardCalcFusions(execPlan, curFuseShimID - 1, nullptr) + numFusion
                     : numFusion;
    }
    // this node is either outside of a shim (shimLastNode == nullptr)
    // or inside of a shim(shimLastNode != nullptr), simply keep on going back...
    else
    {
        numFusedNodes
            = parent ? parent->BackwardCalcFusions(execPlan, curFuseShimID, shimLastNode) : 0;
    }

    return numFusedNodes;
}

size_t PlacementTrace::NumUsedBuffers() const
{
    return usedBuffers.size();
}

void PlacementTrace::Backtracking(ExecPlan& execPlan, int execSeqID)
{
    const auto& execSeq = execPlan.execSeq;

    if((execSeqID < 0) || (curNode != execSeq[execSeqID]))
        throw std::runtime_error("Backtracking error: accessing invalid resource");

    auto node          = execSeq[execSeqID];
    node->placement    = this->isInplace ? rocfft_placement_inplace : rocfft_placement_notinplace;
    node->obIn         = this->inBuf;
    node->obOut        = this->outBuf;
    node->inArrayType  = this->iType;
    node->outArrayType = this->oType;

    // correct array type of callback, since it could be marked as CI during the process
    if(node->scheme == CS_KERNEL_APPLY_CALLBACK)
        node->outArrayType = node->inArrayType = rocfft_array_type_real;
    // correct the obIn to S buffer, not really use the input buffer just for distinguishing
    if(node->scheme == CS_KERNEL_CHIRP)
    {
        node->obIn      = OB_TEMP_BLUESTEIN;
        node->placement = rocfft_placement_inplace;
    }

    // for nodes that uses bluestein buffer
    if(node->obIn == OB_TEMP_BLUESTEIN && node->parent && node->parent->iOffset)
        node->iOffset = node->parent->iOffset;
    if(node->obOut == OB_TEMP_BLUESTEIN && node->parent && node->parent->oOffset)
        node->oOffset = node->parent->oOffset;

    if(execSeqID > 0)
    {
        parent->Backtracking(execPlan, execSeqID - 1);
    }
    else
    {
        // first leaf node must match the root-input
        node->inArrayType = execPlan.rootPlan->inArrayType;
    }
}

std::vector<size_t> AssignmentPolicy::GetEffectiveNodeOutLen(ExecPlan&       execPlan,
                                                             const TreeNode& node)
{
    auto effLen = node.length;

    // R2C kernels change output length, so compensate for that
    if(node.direction == -1)
    {
        // real even: "r2c", or "fused stockham+r2c" (in 3D-even)
        if(node.scheme == CS_KERNEL_R_TO_CMPLX
           || (node.scheme == CS_KERNEL_STOCKHAM && node.ebtype == EmbeddedType::Real2C_POST))
        {
            effLen.front() = effLen.front() + 1; // ex, 32 -> 33
        }
        // odd real: r2c copy tail : CI (but actually is real, node length in real) -> HI (cvt length in cmplx)
        else if(node.scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            effLen.front() = effLen.front() / 2 + 1; // ex, node len = 81 real -> 41 cmplx
        }
    }

    // in 3D even: the inplace sbcc length[1] is modified by /2 + 1, we compensate here
    if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CC && node.parent
       && node.parent->scheme == CS_REAL_3D_EVEN)
    {
        effLen[1] -= 1;
    }

    // transpose kernels swap around their output lengths
    //   (CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY and CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
    //    are fused kernels, perhaps not needed)
    if(node.scheme == CS_KERNEL_TRANSPOSE)
        std::swap(effLen[0], effLen[1]);
    else if(node.scheme == CS_KERNEL_TRANSPOSE_Z_XY
            || node.scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)
    {
        std::swap(effLen[0], effLen[1]);
        std::swap(effLen[1], effLen[2]);
    }
    else if(node.scheme == CS_KERNEL_TRANSPOSE_XY_Z
            || node.scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)
    {
        std::swap(effLen[1], effLen[2]);
        std::swap(effLen[0], effLen[1]);
    }
    // TODO- Check if any other scheme (leaf-node)
    return effLen;
}

// test if rootArrayType == testArrayType,
// if testAryType == rootAryType, return true (since they're definitely equivalent)
// but if root is real or HI, the equivalent internal type could be CI (1-ptr)
// or if root is HP, the equivalent internal type could be CP (2-ptr)
bool AssignmentPolicy::EquivalentArrayType(rocfft_array_type rootAryType,
                                           rocfft_array_type testAryType)
{
    if(rootAryType == rocfft_array_type_real
       || rootAryType == rocfft_array_type_hermitian_interleaved)
        return (testAryType == rootAryType)
               || (testAryType == rocfft_array_type_complex_interleaved);

    if(rootAryType == rocfft_array_type_hermitian_planar)
        return (testAryType == rootAryType) || (testAryType == rocfft_array_type_complex_planar);

    return (testAryType == rootAryType);
}

bool AssignmentPolicy::BufferIsUnitStride(const ExecPlan& execPlan, OperatingBuffer buf)
{
    // temp buffers are unit stride
    if(buf != OB_USER_IN && buf != OB_USER_OUT)
        return true;
    auto stride = (buf == OB_USER_IN) ? execPlan.rootPlan->inStride : execPlan.rootPlan->outStride;
    auto length = (buf == OB_USER_IN) ? execPlan.iLength : execPlan.oLength;
    auto dist   = (buf == OB_USER_IN) ? execPlan.rootPlan->iDist : execPlan.rootPlan->oDist;
    size_t curStride = 1;
    do
    {
        if(stride.front() != curStride)
            return false;
        curStride *= length.front();
        stride.erase(stride.begin());
        length.erase(length.begin());
    } while(!stride.empty());

    // NB: users may input incorrect i/o-dist value for inplace transform
    //     however, when the batch-size is 1, we can simply make it permissive
    //     since the dist is not used in single batch. But note that we still need
    //     to pass the above do-while to ensure all the previous strides are valid.
    return (execPlan.rootPlan->batch == 1) || (curStride == dist);
}

bool AssignmentPolicy::ValidOutBuffer(ExecPlan&           execPlan,
                                      NodeBufTestCacheKey cacheMapKey,
                                      TreeNode&           node,
                                      OperatingBuffer     buffer,
                                      rocfft_array_type   arrayType)
{
    auto cacheMapIter = node_buf_test_cache.find(cacheMapKey);
    if(cacheMapIter != node_buf_test_cache.end())
        return cacheMapIter->second;

    // define a local function.
    auto dataFits = [&execPlan](
                        const TreeNode& node, OperatingBuffer buffer, std::vector<size_t> bufLen) {
        if(node.outputHasPadding)
            return false;

        auto nodeLen = GetEffectiveNodeOutLen(execPlan, node);

        // if node's output is complex and buffer's format is real,
        // adjust output length to be 2x to make the units of
        // comparison match
        bool kernelOutputIsReal = node.scheme == CS_KERNEL_COPY_CMPLX_TO_R;
        bool outBufferIsReal
            = (buffer == OB_USER_OUT && execPlan.rootPlan->outArrayType == rocfft_array_type_real)
              || (buffer == OB_USER_IN && execPlan.rootPlan->inArrayType == rocfft_array_type_real);
        if(outBufferIsReal)
        {
            // special case for "inplace sbcc in real 3D Even"
            if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CC && node.parent
               && node.parent->scheme == CS_REAL_3D_EVEN)
                nodeLen[1] *= 2;
            else if(!kernelOutputIsReal)
                nodeLen.front() *= 2;
        }

        if(BufferIsUnitStride(execPlan, buffer))
        {
            // just check if there's enough space
            return std::accumulate(nodeLen.begin(),
                                   nodeLen.end(),
                                   static_cast<size_t>(1),
                                   std::multiplies<size_t>())
                   <= std::accumulate(bufLen.begin(),
                                      bufLen.end(),
                                      static_cast<size_t>(1),
                                      std::multiplies<size_t>());
        }

        // NB:
        //   TODO- Worth of thinking:
        //   A conservative way is to disallow non-unit-stride CC output to A/B,
        //   But if both A->B are non-uint-s (inS=2 ,outS=2), it's possible to do w/o TempBuf
        //   Somehow I don't see this restriction is a must but still leave a comment here
        // if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        //     return false;

        // ensure that the node's dimensions fit exactly into the
        // buffer's dimensions.  e.g. if the node wants XxYxZ and the
        // buffer is AxZ, this is ok so long as X*Y == A
        for(auto len : nodeLen)
        {
            // not decomposing evenly
            if(bufLen.empty() || bufLen.front() % len != 0)
                return false;
            bufLen.front() /= len;
            if(bufLen.front() == 1)
                bufLen.erase(bufLen.begin());
        }
        return true;
    };

    bool test_result = true;

    // an initial elimination to reject those illegal out-buffer
    if(node.isOutBufAllowed(buffer) == false)
    {
        test_result = false;
    }
    // an initial elimination to reject those illegal out-arrayType
    else if(node.isOutArrayTypeAllowed(arrayType) == false)
    {
        test_result = false;
    }
    // Bluestein and CMPLX buffer must be CI
    else if((buffer == OB_TEMP_BLUESTEIN || buffer == OB_TEMP_CMPLX_FOR_REAL)
            && arrayType != rocfft_array_type_complex_interleaved)
    {
        test_result = false;
    }
    // the only restriction of APPLY_CALLBACK is that it needs to be inplace
    else if(node.scheme == CS_KERNEL_APPLY_CALLBACK)
    {
        test_result = true;
    }
    // More requirement for Bluestein: the 7 component-nodes need to output to bluestein buffer
    // except for the last RES_MUL
    else if(node.IsLastLeafNodeOfBluesteinComponent() && node.scheme != CS_KERNEL_RES_MUL)
    {
        test_result = (buffer == OB_TEMP_BLUESTEIN);
    }
    // if output goes to a temp buffer, that will be dynamically sized
    // to be big enough so it's always ok but if output is in/out, we
    // have to fit into whatever the user gave us
    else if(buffer == OB_USER_IN
            && (!dataFits(node, buffer, execPlan.iLength)
                || !EquivalentArrayType(execPlan.rootPlan->inArrayType, arrayType)))
    {
        test_result = false;
    }
    else if(buffer == OB_USER_OUT
            && (!dataFits(node, buffer, execPlan.oLength)
                || !EquivalentArrayType(execPlan.rootPlan->outArrayType, arrayType)))
    {
        test_result = false;
    }

    node_buf_test_cache[cacheMapKey] = test_result;
    return test_result;
}

static void RecursiveTraverse(TreeNode* node, const std::function<void(TreeNode*)>& func)
{
    func(node);
    for(auto& n : node->childNodes)
        RecursiveTraverse(n.get(), func);
}

bool AssignmentPolicy::CheckAssignmentValid(ExecPlan& execPlan)
{
    int sizeBufIn  = 0;
    int sizeBufOut = 0;
    if(execPlan.rootPlan->placement == rocfft_placement_notinplace)
    {
        sizeBufIn  = execPlan.rootPlan->iDist;
        sizeBufOut = execPlan.rootPlan->oDist;
    }
    else
        sizeBufOut = std::max(execPlan.rootPlan->iDist, execPlan.rootPlan->oDist);

    for(auto& curr : execPlan.execSeq)
    {
        if((curr->obOut == OB_USER_IN && curr->oDist > sizeBufIn)
           || (curr->obOut == OB_USER_OUT && curr->oDist > sizeBufOut))
        {
            // std::cout << "buffer access violation, re-assign" << std::endl;
            return false;
        }

        if(curr->placement == rocfft_placement_inplace)
        {
            const int infact  = curr->inArrayType == rocfft_array_type_real ? 1 : 2;
            const int outfact = curr->outArrayType == rocfft_array_type_real ? 1 : 2;
            if((curr->batch > 1) && (outfact * curr->iDist != infact * curr->oDist))
            {
                // std::cout << "error in dist assignments, re-assign" << std::endl;
                return false;
            }
            for(size_t i = 0; i < curr->inStride.size(); i++)
            {
                if(outfact * curr->inStride[i] != infact * curr->outStride[i])
                {
                    // std::cout << "error in stride assignments, re-assign" << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

void AssignmentPolicy::UpdateWinnerFromValidPaths(ExecPlan& execPlan)
{
    if(winnerCandidates.empty())
        return;

    // std::cout << "total candidates: " << winnerCandidates.size() << std::endl;

    // sort the candidate, front is the best
    std::sort(
        winnerCandidates.begin(), winnerCandidates.end(), [](const auto& lhs, const auto& rhs) {
            // compare numFusedNodes (more is better)
            if(lhs->numFusedNodes > rhs->numFusedNodes)
                return true;
            if(lhs->numFusedNodes < rhs->numFusedNodes)
                return false;

            // if tie, we still choose the one with less buffers
            if(lhs->NumUsedBuffers() < rhs->NumUsedBuffers())
                return true;
            if(lhs->NumUsedBuffers() > rhs->NumUsedBuffers())
                return false;
            // if tie, we still choose the one with more inplace
            if(lhs->numInplace > rhs->numInplace)
                return true;
            if(lhs->numInplace < rhs->numInplace)
                return false;

            // if tie, compare numTypeSwitching (less is better)
            return lhs->numTypeSwitching < rhs->numTypeSwitching;
        });

    for(auto& winner : winnerCandidates)
    {
        // fill the assignment to tree-node from the PlacementTrace path
        winner->Backtracking(execPlan, execPlan.execSeq.size() - 1);

        // assign the stride things. remember to refresh for the internal nodes
        execPlan.rootPlan->RefreshTree();
        // TODO- Next big thing to generalize
        execPlan.rootPlan->AssignParams();

        // Act as a final guard to check the stride and dist
        // Ideally, all the valid-tests were handled in the AssignBuffers,
        // So the first candidate is the result.
        // But some inplace r2c/c2r are tricky and not easy to handle (most of them are dist and stride)
        // This final guard somehow is the "error-detector"...
        // TODO- Eventually we should make the AssignBuffer more robust
        if(CheckAssignmentValid(execPlan))
        {
            // std::cout << "num Fused: " << winner->numFusedNodes;
            // std::cout << ", num IP:" << winner->numInplace;
            // std::cout << std::endl;
            numCurWinnerFusions = winner->numFusedNodes;
            return;
        }
    }
    return;
}

bool AssignmentPolicy::AssignBuffers(ExecPlan& execPlan)
{
    int maxFusions      = execPlan.fuseShims.size();
    numCurWinnerFusions = -1; // no winner yet
    mustUseTBuffer      = false;
    mustUseCBuffer      = false;

    // remember to clear the container either in the beginning or at the end
    winnerCandidates.clear();
    availableBuffers.clear();
    availableArrayTypes.clear();
    node_buf_test_cache.clear();

    // Start from a minimal requirement; // in, out buffer
    availableBuffers.insert(execPlan.rootPlan->obIn);
    availableBuffers.insert(execPlan.rootPlan->obOut);
    if(execPlan.rootPlan->IsRootPlanC2CTransform())
    {
        // For real-transform, USER_IN is always allowed to be modified.
        // For c2c-transform, we should keep USER_IN read-only. So remove it if exists.
        availableBuffers.erase(OB_USER_IN);
    }

    // Insert the valid ArrayTypes to use internally
    rocfft_array_type aliasInType  = execPlan.rootPlan->inArrayType;
    rocfft_array_type aliasOutType = execPlan.rootPlan->outArrayType;

    // if HP/CP -> treated as CP INTERNALLY, else (CI,HI,real) -> treated as CI INTERNALLY
    // if no interleaved in in/out, no need to try, if no planar in in/out, no need to try
    aliasInType  = is_complex_planar(aliasInType) ? rocfft_array_type_complex_planar
                                                  : rocfft_array_type_complex_interleaved;
    aliasOutType = is_complex_planar(aliasOutType) ? rocfft_array_type_complex_planar
                                                   : rocfft_array_type_complex_interleaved;
    availableArrayTypes.insert(aliasInType);
    availableArrayTypes.insert(aliasOutType);

    // look for nodes that imply presence of other buffers (bluestein)
    RecursiveTraverse(execPlan.rootPlan.get(), [this](TreeNode* n) {
        if(n->scheme == CS_KERNEL_CHIRP)
        {
            availableBuffers.insert(OB_TEMP_BLUESTEIN);
            availableArrayTypes.insert(rocfft_array_type_complex_interleaved);
        }
    });

    // First try !
    PlacementTrace dummyRoot;
    dummyRoot.outBuf = execPlan.rootPlan->obIn;
    dummyRoot.oType  = aliasInType;
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // update num-of-winner's-fusions from winnerCandidates list
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
    {
        // we already satisfy the strategy, so don't need to go further
        if(execPlan.assignOptStrategy <= rocfft_optimize_min_buffer)
            return true;
        // we already fulfill all possible fusions
        if(numCurWinnerFusions == maxFusions)
            return true;
    }

    // if we are here:
    // 1. we haven't found a winner (working assignment)
    // 2. we found a assignment but we want to try if it's possible
    //    to have more fusions by adding TEMP buffer
    //    (strategy > rocfft_optimize_min_buffer)
    mustUseTBuffer = true;
    availableBuffers.insert(OB_TEMP);
    dummyRoot.branches.clear();
    winnerCandidates.clear();
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // NB:
    //   in this ABT try, winnerCandidates must contain T-buf (mustUseTBuffer=true)
    //   and it's possible winnerCandidates is empty because there is no new path giving more fusions.
    //   So num-of-winner's-fusions won't be updated, but we may have a winner from prev try
    //   in this case, we should return if the strategy is "balance".
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
    {
        // we already satisfy the strategy, so don't need to go further
        if(execPlan.assignOptStrategy <= rocfft_optimize_balance)
            return true;
        // we already fulfill all possible fusions
        if(numCurWinnerFusions == maxFusions)
            return true;
    }

    // Same as above: if we are here....
    mustUseCBuffer = true;
    availableBuffers.insert(OB_TEMP_CMPLX_FOR_REAL);
    availableArrayTypes.insert(rocfft_array_type_complex_interleaved);
    dummyRoot.branches.clear();
    winnerCandidates.clear();
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // NB:
    //   in this ABTC try, winnerCandidates must contain C-buf (mustUseCBuffer=true)
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
        return true;

    // else, we can't find any valid buffer assignment !
    throw std::runtime_error("Can't find valid buffer assignment with current buffers.");
    return false;
}

void AssignmentPolicy::Enumerate(PlacementTrace*   parent,
                                 ExecPlan&         execPlan,
                                 size_t            curSeqID,
                                 OperatingBuffer   startBuf,
                                 rocfft_array_type startType)
{
    auto& execSeq   = execPlan.execSeq;
    auto& fuseShims = execPlan.fuseShims;

    // Terminal Condition
    // we've done all, check if this path works (matches the rootPlan's out)
    if(curSeqID >= execSeq.size())
    {
        auto endBuf       = execPlan.rootPlan->obOut;
        auto endArrayType = execPlan.rootPlan->outArrayType;

        // the out buf and array type must match
        if(parent->outBuf == endBuf && EquivalentArrayType(endArrayType, parent->oType))
        {
            // we are in the second try (adding T Buffer) but we don't have it in the path:
            // this means we've already tried this path in the previous try.
            if(mustUseTBuffer && parent->usedBuffers.count(OB_TEMP) == 0)
                return;

            // we are in the third try (adding C Buffer) but we don't have it in the path:
            // this means we've already tried this path in the previous try.
            if(mustUseCBuffer && parent->usedBuffers.count(OB_TEMP_CMPLX_FOR_REAL) == 0)
                return;

            // See how many fusions can be done in this path
            int numFusions = parent->BackwardCalcFusions(execPlan, fuseShims.size() - 1, nullptr);
            // skip it if this doesn't outdo the winner of prev. try (prev try = fewer buffers)
            if(numCurWinnerFusions >= numFusions)
                return;

            // set the oType to its original type of RootPlan (for example, change internal-CP to HP)
            parent->oType = endArrayType;

            winnerCandidates.emplace_back(parent);
            // debug
            // parent->Print(*LogSingleton::GetInstance().GetTraceOS());
        }
        return;
    }

    TreeNode* curNode = execSeq[curSeqID];

    // For chirp scheme, the startBuf/AType can differ from parent's, so we don't do this
    // auto startBuf  = parent->outBuf;
    // auto startType = parent->oType;

    if(curNode->scheme == CS_KERNEL_CHIRP)
    {
        // chirp kernel can output to bluestein buffer only,
        // and it doesn't take input buffer, so pass the startBuf buffer to the next node
        // note that input buffer and array type is irrelevant for chirp
        // Create/Push a PlacementTrace for Chirp:
        parent->branches.emplace_back(
            std::make_unique<PlacementTrace>(curNode,
                                             startBuf,
                                             OB_TEMP_BLUESTEIN,
                                             startType,
                                             rocfft_array_type_complex_interleaved,
                                             parent));
        // advance to next
        // NOTE that it is important we propagate startBuf, startType to next node
        Enumerate(parent->branches.back().get(), execPlan, curSeqID + 1, startBuf, startType);
        return;
    }

    // Branch of using inplace, any node dis-alllowing inplace will skip this
    if(curNode->isPlacementAllowed(rocfft_placement_inplace))
    {
        // If buffer is not available (when USER_IN is read-only), skip as well.
        if(availableBuffers.count(startBuf))
        {
            NodeBufTestCacheKey cKey{curSeqID, startBuf, startType};
            if(ValidOutBuffer(execPlan, cKey, *curNode, startBuf, startType))
            {
                // Create/Push a PlacementTrace for an Inplace-Operation (others recurs)
                parent->branches.emplace_back(std::make_unique<PlacementTrace>(
                    curNode, startBuf, startBuf, startType, startType, parent));
                // advance to next
                Enumerate(
                    parent->branches.back().get(), execPlan, curSeqID + 1, startBuf, startType);
            }
        }
    }

    // Branch of using out-of-place, any node dis-alllowing notinplace will skip this
    if(curNode->isPlacementAllowed(rocfft_placement_notinplace))
    {
        // try every available output buffer
        for(auto testOutputBuf : availableBuffers)
        {
            // except for startBuf, since this is a out-of-place try
            if(testOutputBuf == startBuf)
                continue;

            // try every available array type
            for(auto testOutType : availableArrayTypes)
            {
                NodeBufTestCacheKey cKey{curSeqID, testOutputBuf, testOutType};
                if(ValidOutBuffer(execPlan, cKey, *curNode, testOutputBuf, testOutType))
                {
                    // Create/Push a PlacementTrace for OuOfPlace-Operation (others recurs)
                    parent->branches.emplace_back(std::make_unique<PlacementTrace>(
                        curNode, startBuf, testOutputBuf, startType, testOutType, parent));
                    // advance to next
                    Enumerate(parent->branches.back().get(),
                              execPlan,
                              curSeqID + 1,
                              testOutputBuf,
                              testOutType);
                }
            } // end of testing each array type
        } // end of testing each out buffer
    } // end of out-of-place
}
