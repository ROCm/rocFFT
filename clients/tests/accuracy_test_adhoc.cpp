
// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../fft_params.h"

#include "accuracy_test.h"

std::vector<std::vector<size_t>> adhoc_sizes = {
    // sizes that exercise L1D_TRTRT subplan of 2D_RTRT or 3D_TRTRTR
    {1, 220},
    {1, 330},
    {81, 220, 36},

    // L1D_CC subplan of 3D_TRTRTR
    {4, 4, 8192},

    // SBRC 192 with special param
    {192, 192, 192},
    {192, 84, 84},

    // Failure with build_CS_3D_BLOCK_RC
    {680, 128, 128},

    // Large 1D primes that fall above the block threshold (length 262144).
    // Bluestein requires two forwards and one inverse FFTs, and the plan
    // for these sizes breakdown these FFTs either as:
    // L1D_TRTRT (T + STOCKHAM + T + STOCKHAM + T) for lengthBlue <= 4096^2
    // or
    // L1D_TRTRT (T + L1D_CC + STOCKHAM_BL_CC + STOCHMAM_BL_RC + T + STOCKHAM + T)
    // for lengthBlue > 4096^2.
    {196597},
    {25165813},

    // TILE_UNALIGNED type of SBRC 3D ERC
    {98, 98, 98},
};

const static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(adhoc_sizes,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(adhoc_sizes,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

inline auto param_permissive_iodist()
{
    std::vector<std::vector<size_t>> lengths = adhoc_sizes;
    // TODO- for these permissive iodist tests,
    // some 98^3 sizes take too long for the exhaustive search buffer assignments
    // about millions of assignments, thus the program is hung there.
    // So we take this length out from iodist test for now.
    lengths.erase(std::find(lengths.begin(), lengths.end(), std::vector<size_t>{98, 98, 98}));
    lengths.push_back({4});

    std::vector<fft_params> params;
    for(const auto precision : precision_range)
    {
        for(const auto trans_type : trans_type_range)
        {
            for(const auto& types : generate_types(trans_type, place_range, true))
            {
                for(const auto& len : lengths)
                {
                    fft_params param;

                    param.length         = len;
                    param.precision      = precision;
                    param.idist          = 2;
                    param.odist          = 3;
                    param.transform_type = std::get<0>(types);
                    param.placement      = std::get<1>(types);
                    param.itype          = std::get<2>(types);
                    param.otype          = std::get<3>(types);
                    params.push_back(param);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_dist,
                         accuracy_test,
                         ::testing::ValuesIn(param_permissive_iodist()),
                         accuracy_test::TestName);

inline auto param_adhoc_stride()
{
    std::vector<std::vector<size_t>> lengths = adhoc_sizes;
    lengths.push_back({4});

    std::vector<fft_params> params;

    for(const auto precision : precision_range)
    {
        for(const auto& types : generate_types(fft_transform_type_complex_forward,
                                               {fft_placement_inplace, fft_placement_notinplace},
                                               true))
        {
            // 2D with non-contiguous strides and dist
            fft_params param;
            param.length         = {2, 35};
            param.precision      = precision;
            param.idist          = 200;
            param.odist          = 200;
            param.transform_type = fft_transform_type_complex_forward;
            param.nbatch         = 2;
            param.placement      = std::get<1>(types);
            param.itype          = std::get<2>(types);
            param.otype          = std::get<3>(types);
            param.istride        = {90, 2};
            param.ostride        = {90, 2};
            params.push_back(param);
        }

        // test C2R/R2C with non-contiguous higher strides and dist - we
        // want unit stride for length0 so we do the even-length optimization
        for(const auto trans_type :
            {fft_transform_type_real_forward, fft_transform_type_real_inverse})
        {
            for(const auto& types : generate_types(trans_type, {fft_placement_notinplace}, true))
            {
                fft_params param;
                param.length         = {4, 4, 4};
                param.precision      = precision;
                param.idist          = 0;
                param.odist          = 0;
                param.transform_type = trans_type;
                param.nbatch         = 2;
                param.placement      = std::get<1>(types);
                param.itype          = std::get<2>(types);
                param.otype          = std::get<3>(types);
                param.istride        = {16, 4, 1};
                param.ostride        = {16, 4, 1};
                params.push_back(param);

                param.length         = {2, 2, 2};
                param.precision      = precision;
                param.idist          = 0;
                param.odist          = 0;
                param.transform_type = trans_type;
                param.nbatch         = 2;
                param.placement      = std::get<1>(types);
                param.itype          = std::get<2>(types);
                param.otype          = std::get<3>(types);
                param.istride        = {20, 6, 1};
                param.ostride        = {20, 6, 1};
                params.push_back(param);
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_stride,
                         accuracy_test,
                         ::testing::ValuesIn(param_adhoc_stride()),
                         accuracy_test::TestName);

auto adhoc_tokens = {
    "complex_forward_len_512_64_single_ip_batch_3_istride_192_3_CI_ostride_192_3_CI_idist_1_odist_"
    "1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_1024_1024_1024_single_op_batch_1_istride_1048576_1024_1_R_ostride_525312_513_"
    "1_HI_idist_1073741824_odist_537919488_ioffset_0_0_ooffset_0_0",
};

INSTANTIATE_TEST_SUITE_P(adhoc_token,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_token(adhoc_tokens)),
                         accuracy_test::TestName);
