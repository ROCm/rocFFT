// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef FUNCTION_MAP_KEY_H
#define FUNCTION_MAP_KEY_H

#include <array>
#include <tuple>
#include <vector>

#include "compute_scheme.h"
#include "data_descriptor.h"
#include "enum_printer.h"
#include "twiddles.h"

struct KernelConfig
{
    bool                use_3steps_large_twd  = false;
    bool                half_lds              = false;
    bool                direct_to_from_reg    = false;
    bool                intrinsic_buffer_inst = false;
    unsigned int        transforms_per_block  = 0;
    int                 workgroup_size        = 0;
    std::array<int, 2>  threads_per_transform = {0, 0};
    std::vector<size_t> factors               = {0};

    KernelConfig()                    = default;
    KernelConfig(const KernelConfig&) = default;

    KernelConfig(bool                  use_3steps,
                 std::vector<size_t>&& factors,
                 int                   tpb,
                 int                   wgs,
                 std::array<int, 2>&&  tpt,
                 bool                  half_lds              = false,
                 bool                  direct_to_from_reg    = false,
                 bool                  intrinsic_buffer_inst = false)
        : use_3steps_large_twd(use_3steps)
        , half_lds(half_lds)
        , direct_to_from_reg(direct_to_from_reg)
        , intrinsic_buffer_inst(intrinsic_buffer_inst)
        , transforms_per_block(tpb)
        , workgroup_size(wgs)
        , threads_per_transform(tpt)
        , factors(factors)
    {
    }

    KernelConfig& operator=(const KernelConfig&) = default;

    bool operator==(const KernelConfig& rhs) const
    {
        return std::tie(use_3steps_large_twd,
                        half_lds,
                        direct_to_from_reg,
                        intrinsic_buffer_inst,
                        transforms_per_block,
                        workgroup_size,
                        threads_per_transform,
                        factors)
               == std::tie(rhs.use_3steps_large_twd,
                           rhs.half_lds,
                           rhs.direct_to_from_reg,
                           rhs.intrinsic_buffer_inst,
                           rhs.transforms_per_block,
                           rhs.workgroup_size,
                           rhs.threads_per_transform,
                           rhs.factors);
    }

    bool operator<(const KernelConfig& rhs) const
    {
        size_t l_h = std::hash<bool>{}(use_3steps_large_twd);
        size_t r_h = std::hash<bool>{}(rhs.use_3steps_large_twd);
        if(l_h > r_h)
            return true;
        if(l_h < r_h)
            return false;

        l_h = std::hash<bool>{}(half_lds);
        r_h = std::hash<bool>{}(rhs.half_lds);
        if(l_h > r_h)
            return true;
        if(l_h < r_h)
            return false;

        l_h = std::hash<bool>{}(direct_to_from_reg);
        r_h = std::hash<bool>{}(rhs.direct_to_from_reg);
        if(l_h > r_h)
            return true;
        if(l_h < r_h)
            return false;

        l_h = std::hash<bool>{}(intrinsic_buffer_inst);
        r_h = std::hash<bool>{}(rhs.intrinsic_buffer_inst);
        if(l_h > r_h)
            return true;
        if(l_h < r_h)
            return false;

        if(transforms_per_block > rhs.transforms_per_block)
            return true;
        if(transforms_per_block < rhs.transforms_per_block)
            return false;

        if(workgroup_size > rhs.workgroup_size)
            return true;
        if(workgroup_size < rhs.workgroup_size)
            return false;

        if(threads_per_transform > rhs.threads_per_transform)
            return true;
        if(threads_per_transform < rhs.threads_per_transform)
            return false;

        return (factors > rhs.factors);
    }

    std::string Print() const
    {
        std::stringstream ss;
        ss << "KernelConfig: {";

        ss << "3steps: " << (use_3steps_large_twd ? "true" : "false")
           << ", half_lds: " << (half_lds ? "true" : "false")
           << ", direct_reg: " << (direct_to_from_reg ? "true" : "false")
           << ", try_use_buf_inst: " << (intrinsic_buffer_inst ? "true" : "false")
           << ", tpb: " << transforms_per_block << ", wgs: " << workgroup_size << ", tpt: ["
           << threads_per_transform[0] << "," << threads_per_transform[1] << "], factors: [";

        std::string COMMA = "";
        for(auto factor : factors)
        {
            ss << COMMA << factor;
            COMMA = ", ";
        }
        ss << "]";

        ss << "}";

        return ss.str();
    }

    static KernelConfig EmptyConfig()
    {
        static KernelConfig empty;
        return empty;
    }
};

namespace std
{
    // hash function of FMKey requires the definition of hash function of kernel-config
    template <>
    struct hash<KernelConfig>
    {
        size_t operator()(const KernelConfig& config) const noexcept
        {
            size_t h = 0;
            h ^= std::hash<bool>{}(config.use_3steps_large_twd);
            h ^= std::hash<bool>{}(config.half_lds);
            h ^= std::hash<bool>{}(config.direct_to_from_reg);
            h ^= std::hash<bool>{}(config.intrinsic_buffer_inst);
            h ^= std::hash<unsigned int>{}(config.transforms_per_block);
            h ^= std::hash<int>{}(config.workgroup_size);
            for(auto& v : config.threads_per_transform)
                h ^= std::hash<int>{}(v);

            // twiddle_factors defines a TWIDDLES_MAX_RADICES = 8
            // which means the maximal factorization pass is 8
            auto factors_max_len = config.factors;
            factors_max_len.resize(TWIDDLES_MAX_RADICES);

            for(auto& v : factors_max_len)
                h ^= std::hash<size_t>{}(v);
            return h;
        }
    };
}

// Implementing the ToString / FromString (data_descriptor.h)
// for writing-to/reading-from texted-format solution map
template <>
struct ToString<KernelConfig>
{
    std::string print(const KernelConfig& value) const
    {
        std::string      str = "{";
        std::vector<int> tpt = {value.threads_per_transform[0], value.threads_per_transform[1]};

        str += FieldDescriptor<bool>().describe("use_3steps", value.use_3steps_large_twd) + ",";
        str += FieldDescriptor<bool>().describe("half_lds", value.half_lds) + ",";
        str += FieldDescriptor<bool>().describe("dir_reg", value.direct_to_from_reg) + ",";
        str += FieldDescriptor<bool>().describe("buffer_inst", value.intrinsic_buffer_inst) + ",";
        str += FieldDescriptor<unsigned int>().describe("tpb", value.transforms_per_block) + ",";
        str += FieldDescriptor<int>().describe("wgs", value.workgroup_size) + ",";
        str += VectorFieldDescriptor<int>().describe("tpt", tpt) + ",";
        str += VectorFieldDescriptor<size_t>().describe("factors", value.factors);
        str += "}";
        return str;
    }
};

template <>
struct FromString<KernelConfig>
{
    void Get(KernelConfig& ret, std::sregex_token_iterator& current) const
    {
        std::vector<int> tpt;
        size_t           tpb;

        FieldParser<bool>().parse("use_3steps", ret.use_3steps_large_twd, current);
        FieldParser<bool>().parse("half_lds", ret.half_lds, current);
        FieldParser<bool>().parse("dir_reg", ret.direct_to_from_reg, current);
        FieldParser<bool>().parse("buffer_inst", ret.intrinsic_buffer_inst, current);
        FieldParser<size_t>().parse("tpb", tpb, current);
        FieldParser<int>().parse("wgs", ret.workgroup_size, current);
        VectorFieldParser<int>().parse("tpt", tpt, current);
        VectorFieldParser<size_t>().parse("factors", ret.factors, current);

        ret.transforms_per_block     = tpb;
        ret.threads_per_transform[0] = tpt[0];
        ret.threads_per_transform[1] = tpt[1];
    }
};

// length, precision, scheme are theose fundemantal information of a kernel;
// SBRC_TRANS is also neccessary for SBRC or SBRC_3D, but for non-SBRC, it is just NONE
// And the newly added KernerlConfig is the key to supporting the "multi-configurations".
// KernelConfig denotes what parameters we can alter to "generate and tune" a kernel
//
// NB:
//    Since we didn't have the KernelConfig before, so, when getting the default kernels
//    from the function_pool, the kernel_config "variable" would be a default EmptyConfig().
//    But actually, the config is defined in the kernel-generator.py, so we are still able to
//    know how the "EmptyConfig" can be mapped to a non-empty config (in kernel-gerator.py)
//    (And that is what exactly "fuction_pool::insert_default_entry()" and
//                               "function_pool::get_actual_key()"" is doing
//
// TODO:
//    eventually, it would be better to implement the FMKey to struct for better readibility
//
using FMKey = std::tuple<std::array<size_t, 2>,
                         rocfft_precision,
                         ComputeScheme,
                         SBRC_TRANSPOSE_TYPE,
                         KernelConfig>;

static inline FMKey fpkey(size_t              length,
                          rocfft_precision    precision,
                          ComputeScheme       scheme        = CS_KERNEL_STOCKHAM,
                          SBRC_TRANSPOSE_TYPE transpose     = NONE,
                          KernelConfig        kernel_config = KernelConfig::EmptyConfig())
{
    return {{length, 0}, precision, scheme, transpose, kernel_config};
}

static inline FMKey fpkey(size_t              length1,
                          size_t              length2,
                          rocfft_precision    precision,
                          ComputeScheme       scheme        = CS_KERNEL_2D_SINGLE,
                          SBRC_TRANSPOSE_TYPE transpose     = NONE,
                          KernelConfig        kernel_config = KernelConfig::EmptyConfig())
{
    return {{length1, length2}, precision, scheme, transpose, kernel_config};
}

// add an alternative kernel with different kernel config from base FMKey
static FMKey get_alternative_FMKey(const FMKey& base_FMKey, const KernelConfig& alt_config)
{
    const auto&               lengthVec = std::get<0>(base_FMKey);
    const rocfft_precision    precision = std::get<1>(base_FMKey);
    const ComputeScheme       scheme    = std::get<2>(base_FMKey);
    const SBRC_TRANSPOSE_TYPE trans     = std::get<3>(base_FMKey);

    return {lengthVec, precision, scheme, trans, alt_config};
}

static void GetKernelToken(const FMKey& key, std::string& min_token)
{
    const auto&            lengthVec = std::get<0>(key);
    const rocfft_precision precision = std::get<1>(key);
    const ComputeScheme    scheme    = std::get<2>(key);

    min_token = "kernel";

    min_token += "_len";
    min_token += std::to_string(lengthVec[0]);
    if(scheme == CS_KERNEL_2D_SINGLE)
        min_token += "x" + std::to_string(lengthVec[1]);

    min_token += "_" + PrintPrecision(precision);
    min_token += "_" + PrintKernelSchemeAbbr(scheme);

    // NB: KernelToken is used when tuning the kernel configuration,
    //     But when we try different setting of TPB, the SBRCTransType
    //     would not be the same value. So we should not keep the SBRCTransType
    //     in the token, and all the SBRC kernels in that solution-vec may have
    //     specify the real type.
    // const SBRC_TRANSPOSE_TYPE transType = std::get<3>(key);
    // min_token += "_" + PrintSBRCTransposeType(transType);
}

template <>
struct ToString<FMKey>
{
    std::string print(const FMKey& value) const
    {
        std::string         str     = "{";
        auto                len     = std::get<0>(value);
        std::vector<size_t> lengths = {len[0], len[1]};

        str += VectorFieldDescriptor<size_t>().describe("lengths", lengths) + ",";
        str += FieldDescriptor<std::string>().describe("precision",
                                                       PrintPrecision(std::get<1>(value)))
               + ",";
        str += FieldDescriptor<std::string>().describe("scheme", PrintScheme(std::get<2>(value)))
               + ",";
        str += FieldDescriptor<std::string>().describe("sbrc_trans",
                                                       PrintSBRCTransposeType(std::get<3>(value)))
               + ",";
        str += FieldDescriptor<KernelConfig>().describe("kernelConfig", std::get<4>(value));
        str += "}";
        return str;
    }
};

template <>
struct FromString<FMKey>
{
    void Get(FMKey& ret, std::sregex_token_iterator& current) const
    {
        std::vector<size_t> len;
        std::string         precStr, schemeStr, sbrcTransStr;
        KernelConfig        config;

        VectorFieldParser<size_t>().parse("lengths", len, current);
        FieldParser<std::string>().parse("precision", precStr, current);
        FieldParser<std::string>().parse("scheme", schemeStr, current);
        FieldParser<std::string>().parse("sbrc_trans", sbrcTransStr, current);
        FieldParser<KernelConfig>().parse("kernelConfig", config, current);

        ret = {{len[0], len[1]},
               StrToPrecision(precStr),
               StrToComputeScheme(schemeStr),
               StrToSBRCTransType(sbrcTransStr),
               config};
    }
};

static FMKey              EmptyFMKey    = {};
static std::vector<FMKey> EmptyFMKeyVec = {};

struct SimpleHash
{
    size_t operator()(const FMKey& p) const noexcept
    {
        size_t h = 0;
        for(auto& v : std::get<0>(p))
            h ^= std::hash<int>{}(v);
        h ^= std::hash<rocfft_precision>{}(std::get<1>(p));
        h ^= std::hash<ComputeScheme>{}(std::get<2>(p));
        h ^= std::hash<SBRC_TRANSPOSE_TYPE>{}(std::get<3>(p));
        h ^= std::hash<KernelConfig>{}(std::get<4>(p));

        return h;
    }
};

#endif
