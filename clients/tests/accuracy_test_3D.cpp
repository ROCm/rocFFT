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

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "../client_utils.h"

#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

// Set parameters

// TODO: 512, 1024, 2048 make the tests take too long; re-enable when
// test speed is improved.
static std::vector<size_t> pow2_range = {4, 8, 16, 32, 128, 256};
// SBCC+SBRC as a sub-node of a 3D TRTRTR
std::vector<std::vector<size_t>> pow2_adhoc = {{4, 4, 8192}};

static std::vector<size_t> pow3_range = {3, 9, 27, 81, 243};

static std::vector<size_t> pow5_range = {5, 25, 125};

static std::vector<size_t> prime_range = {7, 11, 13, 17, 19, 23, 29};

static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}};

static std::vector<std::vector<size_t>> vpow2_range = {pow2_range, pow2_range, pow2_range};
INSTANTIATE_TEST_SUITE_P(pow2_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vpow2_range,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

static std::vector<std::vector<size_t>> vpow3_range = {pow3_range, pow3_range, pow3_range};
INSTANTIATE_TEST_SUITE_P(pow3_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vpow3_range,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

static std::vector<std::vector<size_t>> vpow5_range = {pow5_range, pow5_range, pow5_range};
INSTANTIATE_TEST_SUITE_P(pow5_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vpow5_range,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

static std::vector<std::vector<size_t>> vprime_range = {prime_range, prime_range, prime_range};
INSTANTIATE_TEST_SUITE_P(prime_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vprime_range,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

static std::vector<std::vector<size_t>> vmix_range = {pow2_range, pow3_range, prime_range};

INSTANTIATE_TEST_SUITE_P(mix_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vmix_range,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

// Test combinations of SBRC sizes, plus a non-SBRC size (10) to
// exercise fused SBRC+transpose kernels.
static std::vector<size_t> sbrc_range       = {64, 81, 100, 200, 10, 128, 256};
auto                       vsbrc_range      = {sbrc_range, sbrc_range, sbrc_range};
static std::vector<size_t> sbrc_batch_range = {2, 1};
INSTANTIATE_TEST_SUITE_P(sbrc_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(vsbrc_range,
                                                             precision_range,
                                                             sbrc_batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);
