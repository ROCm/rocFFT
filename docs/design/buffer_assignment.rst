Buffer assignment design document for rocFFT
============================================

Copyright and Disclaimer
------------------------

DISCLAIMER

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc.  Other product names
used in this publication are for identification purposes only and may
be trademarks of their respective companies.

Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights
reserved.

Summary
-------

Buffer assignment in rocFFT is the work of coordinating the input
and output buffers of each step in a rocFFT plan.

Observations
------------

Some observations can be made about the FFT planning and buffer
assignment process:

1. Buffer assignment begins after the plan structure is decided.
   This means all of the node schemes, as well as input lengths and
   input/output strides are known.

   Note that at this time, output lengths are not directly known.

2. The first child of any non-leaf node must read from its parent's
   input, and the last child must write to its parent's output.

3. The input of any other child node must be the same as the output
   of its preceding sibling.

   There is one exception: a CS_KERNEL_CHIRP node is only used to
   populate the Bluestein temporary buffer, and can essentially be
   ignored during buffer processing as it does not actually read
   input.

4. The top-level node in the tree must read from the user-defined
   input buffer, and write to the user-defined output buffer.  These
   buffers will be the same for in-place transforms.

5. When deciding buffer assignments for a node, only the output
   buffer for nodes besides the last requires actual decision making.
   The input buffer follows either from the top-level input, a
   preceding sibling, or a parent node's input.  The last node's
   output must be the user-defined output.

6. During buffer assignment, some number of buffers are available.
   At minimum, we have the user input and output buffers (which may
   be the same for in-place transforms) whose sizes are defined by
   the user.

   Zero or more temporary buffers may be needed, whose sizes are
   dynamic and can be as big as necessary for the transform to
   succeed.

7. Some choices of output buffers are clearly invalid.  For example:

   * Transpose nodes must always be out-of-place (i.e. input buffer
     cannot equal output buffer).

   * Some internal kernels only support interleaved formats as their
     input or output.  For example, the input of a copy-kernel (like
     COPY_CMPLX_TO_R or COPY_CMPLX_TO_HERM) must be interleaved.

   * Internal temp buffers are allocated contigously, so they can be
     used on both planar and interlevead formats.  This is not always
     true for user-provided buffers.  An obvious example of this
     planar data: users typically create these using two buffers.

   * A node cannot write data to a buffer if the buffer is too small
     for that data.  This really only applies to user input/output
     buffers, as temp buffers are always made large enough.

Solution
--------

We implement a decision function that determines whether a buffer
assignment is valid based on the observations above.

Buffer assignment should do an exhaustive search through the space of
possible buffer assignments for the tree, calling the decision
function for each potential choice.  If we arrive at the end of the
tree and all assignments are valid, then the buffer assignment
operation is complete.

Returning the first valid buffer assignment found is a simple
solution.  However, not all valid buffer assignments are equal in
terms of memory usage and/or performance: some buffer assignments
allow more kernel fusions and/or use more inplace kernels.  This
implies that we should keep all valid assignment candidates in a list
and subsequently return the "best" one.

The first pass of buffer assignment shall attempt to assign buffers
starting with just the user input buffer (and output buffer, if
distinct) and a Bluestein temp buffer (if the plan requires it), but
without any other temp buffers.

If that first pass is not successful, we retry with one temp buffer
added to the list of available buffers.  If that is still not
successful, we retry with a second temp buffer.

Implementation
--------------

A Structure Storing A Try
^^^^^^^^^^^^^^^^^^^^^^^^^^

We store our current assignment try in a tree-like structure.  We
don't assign to the tree-node directly since there could be many valid
assignment paths for one plan.  Once we determine the best assignment
path from this tree, we fill the assignment back to the real
tree-node.

.. code-block:: cpp

  struct PlacementTrace
  {
      TreeNode* referedNode;
      Buffer inBuffer, outBuffer;
      ArrayType inArrayType, outArrayType;

      // each branch stands for a possible try on the next node
      vector<PlacementTrace*> branches;

      // a parent pointer to backtracing
      PlacementTrace* parent;

      // propagate these values from the root to leaves
      size_t numFusedNodes;
      size_t numInplace;
      size_t numTypwSwithching;
  }


Exhaustive Search
^^^^^^^^^^^^^^^^^

All possible assignments on each node are attempted.  There are
several limitations on each node that allow us to reject many illegal
assignments and prevent the assignment tree from growing
exponentially.  For example, SBRC and transpose kernels can only be
done using out-of-place buffers.

The exhaustive search is implemented in pseudocode like:

.. code-block:: cpp

   // ------------------------------------------------------------------------------------
   // Recursive function enumrates all the possible assignments
   // Returns a sub-tree, starting from execSeq[curSeqID], with input startBuf & startAType
   // ------------------------------------------------------------------------------------
   Function: void Enumerate(PlacementTrace* parent, ExecPlan, curSeqID, startBuf, startAType)
   // for terminal condition:
   - if curSeqID is the last nodes
     - if the end buffer and array-type fit the root-plan setting
       - calculate the number of eligible kernel-fusions.
       - add this candidate to the winnerCandidates list.
       - finish this path, return

   // not terminal condition:
   // add a single assignment on current node and append to parent's branches
   - if current node->isPlacementAllowed(inplace)
     // add a branch which uses inplace (if allowed) on this node and test validity
     - if ValidOutBuffer(execPlan, *curNode, startBuf, startType)
       - append an assignIP = PlacementTrace(curNode, startBuf, startBuf, startType, startType, parent)
       - call Enumerate(IPAssign, execPlan, curSeqID + 1, startBuf, startType);

   - if current node->isPlacementAllowed(out-of-place)
     // add branches which use out-of-place (if allowed) on this node and test validity
     - for each testOutputBuf in the availableBuffers set, (where testOutputBuf != startBuf)
       - if ValidOutBuffer(execPlan, *curNode, testOutputBuf, testOutType)
         - append an assignOP = PlacementTrace(curNode, startBuf, testOutputBuf, startType, testOutType, parent)
         - call Enumerate(OPAssign, execPlan, curSeqID + 1, testOutputBuf, testOutType);

   // --------------------------------------------------------
   // Decision maker: choose the best one from all candidates
   // This function is a sorting function, pick the first one
   // --------------------------------------------------------
   Function: void ValidPathExists(ExecPlan)
   - if winnerCandidates is empty, simply return false
   - using std::sort, sort by:
     // the one can fuse more kernels is better
     - lhs->numFusedNodes > rhs->numFusedNodes ?
     // if tie, compare inplace kernels, more is better
     - lhs->numInplace > rhs->numInplace ?
     // if tie, compare the times of switching-array-type, less is better
     - lhs->numTypeSwitching < rhs->numTypeSwitching ?

   - pick the first one, and do the Backtracking()
     - fill-in the assignment back to the real tree-nodes

   // ---------------------------------------------------------
   // Top-level function that assigns buffers on the root plan
   // ---------------------------------------------------------
   Function: void AssignBuffers(ExecPlan)
   - add rootPlan in/out buffer to availableBuffers set
     - Note: For C2C out-of-place, we can't add USER_IN to the set to prevent it from being modified.
   - add rootPlan in/out array-type to availableArrayTypes set
   - add OB_TEMP_BLUESTEIN to availableBuffers set, if plan uses Bluestein
   - initialize a winnerCandidates list to save all valid results.
   - initialize a dummyRoot of PlacementTrace as tree root, this dummyRoot pretends it's a parent of the first node (in execSeq).
     So dummyRoot.outBuf = rootPlan->obIn, and dummyRoot.oType = rootPlan->inArrayType

   // The 1st round try
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // The 2nd round try
   - add OB_TEMP to availableBuffers
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // The last round try
   - add OB_TEMP_CMPLX_FOR_REAL to availableBuffers
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // Failed
   - if not found, throw exception.

Decision Function and Output Lengths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Much of the remaining complexity lies in the ValidOutBuffer()
decision function mentioned above.

Output lengths are not always trivially knowable from input lengths in
some cases:

* Padding introduced in some plans such as L1D_TRTRT, 2D_RTRT, or
  REAL_2D_EVEN.  This is partially alleviated by setting a
  "outputHasPadding" flag before the buffer assignment process.

* R2C/C2R handling of Hermitian redundancy.  Without explicit output
  length information, it's not possible to make a decision on buffers.
  In order to handle this, we calculate the real buffer length of the
  root plan itself, depending on if it's R2C or C2R, IP or OP, which
  gives us the explicit length of in/out buffer size.

Since the exhaustive search is a depth-first-search, so when we go back
to the upper nodes and proceed to another branch, the node-vs-buffer-test
in the deeper nodes could be repeated, so we can put the test results in
a cache to make a slight optimization.

Fusions
^^^^^^^

Kernel-fusion is essential for improving performance.  Unfortunately
fusion depends heavily on buffer assignment.  Two (or more) kernels
can be fused into one kernel only when the resulting buffer assignment
remains valid.

To maximise kernel fusion, we also implement a FuseShim framework. A
FuseShim class is a container/shell indicating that there is a
potentially-fusable kernel-fusion.  Each FuseShim defines its own
requirements to fulfill the fusion, including the expected buffer
assignment.

During the buffer assignment process, we can use the test function to
get the final number of the achievable kernel fusions.  This number
plays the most important role when making the final decision: we
always pick the one which can fuse the most kernels.

Future Work
-----------

Strides
^^^^^^^

Currently, rocFFT does not guarantee that strides on user buffers are
respected if temporary data is written to those buffers.

With this implementation, it would be simpler to begin enforcing such
a guarantee.

Enforcing Read-only Input
^^^^^^^^^^^^^^^^^^^^^^^^^

rocFFT may currently overwrite user input buffers for out-of-place
real-transforms (not C2C-transform).  Although we've documented this
behaviour and it is common practice in other libraries, it might still
be unintuitive for some users.

If we ever wanted to start guaranteeing that user input is left
unmodified, this buffer assignment implementation would make that work
trivial - only the decision function needs to be made aware of this
policy change, and buffer assignment will work fine.

However, we may need to introduce yet another temp buffer, since we'd
be taking away a potential work space from existing plans.

Flexibility Between Minimizing Memory or Maximizing Fusions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can't always expect there is a perfect assignment that maximises
kernel fusions while also minimising temporary buffers.  In some
cases, these two goals are contradictory: if we choose an assignment
using minimal buffers, we may loose the oppurtunity to fuse more
kernels.  On the other hand, if we are allowed to use more memory, we
have more buffers available for out-of-place kernel-fusions.

With this implementation, it is possible to introduce an optimization
strategy option to users.

For example, if the memory usage is the main concern of users, we can
return the assignment with least buffer usage.  Otherwise, we return
the result which maximizes the kernel fusions regardless of the memory
consumption.

Make C Buffer as Temp2 Buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no reason to limit the "C" buffer to real-transforms only.
We can make the C buffer as another generic temporary buffer throughout;
this can also avoid any confusion about the purpose of C and T.
