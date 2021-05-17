// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node.h"

// TODO:
//   - better data structure, and more elements for non pow of 2
//   - validate corresponding functions existing in function pool or not
TreeNode::Map1DLength const TreeNode::map1DLengthSingle
    = {{8192, 64}, // pow of 2: CC (64cc + 128rc)
       {16384, 64}, //          CC (64cc + 256rc) // 128x128 no faster
       {32768, 128}, //         CC (128cc + 256rc)
       {65536, 256}, //         CC (256cc + 256rc)
       {131072, 64}, //         CRT(64cc + 2048 + transpose)
       {262144, 64}, //         CRT(64cc + 4096 + transpose)
       {6561, 81}, // pow of 3: CC (81cc + 81rc)
       {10000, 100}, // mixed:  CC (100cc + 100rc)
       {40000, 200}}; //        CC (200cc + 200rc)

TreeNode::Map1DLength const TreeNode::map1DLengthDouble
    = {{4096, 64}, // pow of 2: CC (64cc + 64rc)
       {8192, 64}, //           CC (64cc + 128rc)
       {16384, 128}, //         CC (128cc + 128rc) // faster than 64x256
       {32768, 128}, //         CC (128cc + 256rc)
       {65536, 256}, //         CC (256cc + 256rc) // {65536, 64}
       {131072, 64}, //         CRT(64cc + 2048 + transpose)
       {6561, 81}, // pow of 3: CC (81cc + 81rc)
       {2500, 50}, // mixed:    CC (50cc + 50rc)
       {10000, 100}, //         CC (100cc + 100rc)
       {40000, 200}}; //        CC (200cc + 200rc)
