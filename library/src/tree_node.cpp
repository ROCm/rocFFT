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
TreeNode::Map1DLength const TreeNode::map1DLengthSingle = {{8192, 64}, // pow of 2
                                                           {16384, 64},
                                                           {32768, 128},
                                                           {65536, 256},
                                                           {131072, 64},
                                                           {262144, 64},
                                                           {6561, 81}, // pow of 3
                                                           {10000, 100}, // mixed
                                                           {40000, 200}};

TreeNode::Map1DLength const TreeNode::map1DLengthDouble = {{4096, 64}, // pow of 2
                                                           {8192, 64},
                                                           {16384, 64},
                                                           {32768, 128},
                                                           {65536, 64},
                                                           {131072, 64},
                                                           {6561, 81}, // pow of 3
                                                           {2500, 50}, // mixed
                                                           {10000, 100},
                                                           {40000, 200}};
