// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <functional>
#include <iostream>
#include <thread>

using namespace std::placeholders;

#include "../../shared/concurrency.h"
#include "../../shared/environment.h"
#include "../../shared/work_queue.h"
#include "function_pool.h"
#include "rtc_cache.h"
#include "rtc_realcomplex_gen.h"
#include "rtc_stockham_gen.h"

#include "device/kernel-generator-embed.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif
namespace fs = std::filesystem;

struct WorkItem
{
    std::string      kernel_name;
    kernel_src_gen_t generate_src;
};
typedef WorkQueue<WorkItem> CompileQueue;

// call supplied function with exploded out combinations of
// direction, placement, array types, unitstride-ness, callbacks
void stockham_combo(ComputeScheme             scheme,
                    FFTKernel                 kernel,
                    std::function<void(int,
                                       rocfft_result_placement,
                                       rocfft_array_type,
                                       rocfft_array_type,
                                       EmbeddedType,
                                       SBRC_TRANSPOSE_TYPE,
                                       DirectRegType,
                                       IntrinsicAccessType,
                                       int,
                                       int,
                                       bool,
                                       bool)> func)
{
    // unit stride is the common case, default to that
    std::vector<bool>                    unitstride_range = {true};
    std::vector<rocfft_result_placement> placements       = {rocfft_placement_notinplace};
    std::vector<EmbeddedType>            ebtypes          = {EmbeddedType::NONE};
    std::vector<SBRC_TRANSPOSE_TYPE>     sbrc_trans_types = {SBRC_TRANSPOSE_TYPE::NONE};
    std::vector<DirectRegType>           dir_reg_types    = {FORCE_OFF_OR_NOT_SUPPORT};
    std::vector<IntrinsicAccessType>     intrinsic_modes  = {DISABLE_BOTH};
    // SBCC can be used with or without large twd.  Large twd may be
    // base 4, 5, 6, 8.  Base 4 is unused since it's only useful up
    // to 4k lengths, which we already have single kernels for.  Base
    // 8 can be 2 or 3 steps; other bases are always 3 step.
    std::vector<std::array<int, 2>> base_steps = {{0, 0}, {5, 3}, {6, 3}, {8, 2}, {8, 3}};

    switch(scheme)
    {
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    {
        placements.push_back(rocfft_placement_inplace);
        // SBCC is never unit stride
        unitstride_range = {false};

        // if no dir-to-reg support, then we don't have intrinsic buffer RW,
        // and only force_off_or_not_support for dir2reg. Else, we have all possibilities
        // (even force_off_or_not_support is still included)
        if(kernel.direct_to_from_reg)
        {
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_BOTH);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_LOAD_ONLY);
        }
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    {
        base_steps.resize(1);
        // SBCR is never unit stride
        unitstride_range = {false};
        ebtypes.push_back(EmbeddedType::C2Real_PRE);

        // if no dir-to-reg support, then we don't have intrinsic
        // buffer RW, and only force_off_or_not_support for
        // dir2reg. Else, SBCR only allows "both" or "none" for
        // intrinsic.
        if(kernel.direct_to_from_reg)
        {
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_BOTH);
        }
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
    {
        // SBRC allows direct to/from reg but does not support
        // intrinsic
        if(kernel.direct_to_from_reg)
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);

        // SBRC is never unit stride
        unitstride_range = {false};

        base_steps.resize(1);
        // All SBRCs have TILE_UNALIGNED
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::TILE_UNALIGNED);
        // Finish SBRC-2D
        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
            break;
        // All 3D SBRCs have TILE_ALIGNED, but "NO" SBRC_TRANSPOSE_TYPE::NONE
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::TILE_ALIGNED);
        sbrc_trans_types.erase(sbrc_trans_types.begin());
        // Finish ERC
        if(scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
            break;
        // DIAGNAL Transpose
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::DIAGONAL);

        break;
    }
    case CS_KERNEL_STOCKHAM:
    {
        base_steps.resize(1);
        unitstride_range = {true, false};
        placements.push_back(rocfft_placement_inplace);
        ebtypes.push_back(EmbeddedType::Real2C_POST);
        ebtypes.push_back(EmbeddedType::C2Real_PRE);

        // SBRR always uses TRY_ENABLE_IF_SUPPORT, and does not
        // support intrinsic parameter.
        dir_reg_types = {DirectRegType::TRY_ENABLE_IF_SUPPORT};
        break;
    }
    default:
        // throw std::runtime_error("unsupported scheme in stockham_combo aot_rtc");
        // since it is not possible that we are here,
        // so directly return is fine which means do nothing
        return;
    }

    for(auto direction : {-1, 1})
    {
        for(auto placement : placements)
        {
            for(auto inArrayType : {rocfft_array_type_complex_interleaved})
            {
                for(auto outArrayType : {rocfft_array_type_complex_interleaved})
                {
                    // inplace requires same array types
                    if(placement == rocfft_placement_inplace && inArrayType != outArrayType)
                        continue;
                    for(auto unitstride : unitstride_range)
                    {
                        for(auto base_step : base_steps)
                        {
                            for(auto ebtype : ebtypes)
                            {
                                for(auto sbrc_trans_type : sbrc_trans_types)
                                {
                                    for(auto dir_reg_type : dir_reg_types)
                                    {
                                        for(auto intrinsic : intrinsic_modes)
                                        {
                                            for(auto callback : {true, false})
                                            {
                                                func(direction,
                                                     placement,
                                                     inArrayType,
                                                     outArrayType,
                                                     ebtype,
                                                     sbrc_trans_type,
                                                     dir_reg_type,
                                                     intrinsic,
                                                     base_step[0],
                                                     base_step[1],
                                                     unitstride,
                                                     callback);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void build_stockham_function_pool(CompileQueue& queue)
{
    // build everything in the function pool
    function_pool& fp = function_pool::get_function_pool();

    // scaling Stockham kernels are always built at runtime
    const bool enable_scaling = false;

    for(const auto& i : fp.get_map())
    {
        // we only want to compile kernels explicitly marked for AOT RTC
        if(!i.second.aot_rtc)
            continue;

        auto length1D = std::get<0>(i.first)[0];
        // auto length2D            = std::get<0>(i.first)[1];
        auto                      precision = std::get<1>(i.first);
        auto                      scheme    = std::get<2>(i.first);
        std::vector<unsigned int> factors;
        std::copy(i.second.factors.begin(), i.second.factors.end(), std::back_inserter(factors));

        stockham_combo(scheme,
                       i.second,
                       [=, &queue](int                     direction,
                                   rocfft_result_placement placement,
                                   rocfft_array_type       inArrayType,
                                   rocfft_array_type       outArrayType,
                                   EmbeddedType            ebtype,
                                   SBRC_TRANSPOSE_TYPE     sbrc_trans_type,
                                   DirectRegType           dir_reg_type,
                                   IntrinsicAccessType     intrinsic,
                                   int                     ltwd_base,
                                   int                     ltwd_step,
                                   bool                    unitstride,
                                   bool                    callbacks) {
                           // intrinsic mode require non-callback and enable dir_reg
                           if((callbacks || dir_reg_type == FORCE_OFF_OR_NOT_SUPPORT)
                              && (intrinsic != IntrinsicAccessType::DISABLE_BOTH))
                               return;

                           // for SBRR kernels, only do embedded pre/post processing for
                           // unit-stride, since that's all we'd use it for.
                           //
                           // we currently also only do this processing for even lengths.
                           //
                           // callbacks are not expected to be needed either, since
                           // either a higher dimension FFT or APPLY_CALLBACK would
                           // be expected to run before/after.
                           if(ebtype != EmbeddedType::NONE)
                           {
                               if((scheme == CS_KERNEL_STOCKHAM && !unitstride) || length1D % 2 != 0
                                  || callbacks)
                                   return;
                           }

                           auto kernel_name = stockham_rtc_kernel_name(scheme,
                                                                       length1D,
                                                                       0,
                                                                       0,
                                                                       direction,
                                                                       precision,
                                                                       placement,
                                                                       inArrayType,
                                                                       outArrayType,
                                                                       unitstride,
                                                                       ltwd_base,
                                                                       ltwd_step,
                                                                       false,
                                                                       ebtype,
                                                                       dir_reg_type,
                                                                       intrinsic,
                                                                       sbrc_trans_type,
                                                                       callbacks,
                                                                       enable_scaling);
                           std::function<std::string(const std::string&)> generate_src
                               = [=](const std::string& kernel_name) -> std::string {
                               StockhamGeneratorSpecs specs{
                                   factors,
                                   {},
                                   {static_cast<unsigned int>(precision)},
                                   static_cast<unsigned int>(i.second.workgroup_size),
                                   PrintScheme(scheme)};
                               specs.threads_per_transform = i.second.threads_per_transform[0];
                               specs.half_lds              = i.second.half_lds;
                               specs.direct_to_from_reg    = i.second.direct_to_from_reg;
                               return stockham_rtc(specs,
                                                   specs,
                                                   nullptr,
                                                   kernel_name,
                                                   scheme,
                                                   direction,
                                                   precision,
                                                   placement,
                                                   inArrayType,
                                                   outArrayType,
                                                   unitstride,
                                                   ltwd_base,
                                                   ltwd_step,
                                                   false,
                                                   ebtype,
                                                   dir_reg_type,
                                                   intrinsic,
                                                   sbrc_trans_type,
                                                   callbacks,
                                                   enable_scaling);
                           };
                           queue.push({kernel_name, generate_src});
                       });
    }
}

void build_realcomplex(CompileQueue& queue)
{
    for(auto precision : {rocfft_precision_single, rocfft_precision_double})
    {
        for(bool planar : {true, false})
        {
            // build even-length kernels, as they're commonly used
            for(auto scheme : {CS_KERNEL_R_TO_CMPLX, CS_KERNEL_CMPLX_TO_R})
            {
                for(size_t dim : {1, 2, 3})
                {
                    for(bool Ndiv4 : {true, false})
                    {
                        // standalone even-length kernels may be
                        // first/last in the plan, so allow for
                        // callbacks
                        for(bool enable_callbacks : {true, false})
                        {
                            // r2c may have planar output, c2r may have planar input
                            auto inArrayType  = (scheme == CS_KERNEL_CMPLX_TO_R && planar)
                                                    ? rocfft_array_type_complex_planar
                                                    : rocfft_array_type_complex_interleaved;
                            auto outArrayType = (scheme == CS_KERNEL_R_TO_CMPLX && planar)
                                                    ? rocfft_array_type_complex_planar
                                                    : rocfft_array_type_complex_interleaved;

                            RealComplexEvenSpecs specs{{scheme,
                                                        dim,
                                                        precision,
                                                        inArrayType,
                                                        outArrayType,
                                                        enable_callbacks,
                                                        false},
                                                       Ndiv4};
                            auto kernel_name = realcomplex_even_rtc_kernel_name(specs);
                            std::function<std::string(const std::string&)> generate_src
                                = [=](const std::string& kernel_name) -> std::string {
                                return realcomplex_even_rtc(kernel_name, specs);
                            };
                            queue.push({kernel_name, generate_src});
                        }
                    }
                }
            }
            for(auto scheme : {CS_KERNEL_R_TO_CMPLX_TRANSPOSE, CS_KERNEL_TRANSPOSE_CMPLX_TO_R})
            {
                // r2c may have planar output, c2r may have planar input
                auto inArrayType  = (scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R && planar)
                                        ? rocfft_array_type_complex_planar
                                        : rocfft_array_type_complex_interleaved;
                auto outArrayType = (scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE && planar)
                                        ? rocfft_array_type_complex_planar
                                        : rocfft_array_type_complex_interleaved;

                RealComplexEvenTransposeSpecs specs{{scheme,
                                                     static_cast<size_t>(1),
                                                     precision,
                                                     inArrayType,
                                                     outArrayType,
                                                     false,
                                                     false}};
                auto kernel_name = realcomplex_even_transpose_rtc_kernel_name(specs);
                std::function<std::string(const std::string&)> generate_src
                    = [=](const std::string& kernel_name) -> std::string {
                    return realcomplex_even_transpose_rtc(kernel_name, specs);
                };
                queue.push({kernel_name, generate_src});
            }
        }
    }
}

void build_apply_callback(CompileQueue& queue)
{
    for(auto precision : {rocfft_precision_single, rocfft_precision_double})
    {
        auto kernel_name = apply_callback_rtc_kernel_name(precision);
        std::function<std::string(const std::string&)> generate_src
            = [=](const std::string& kernel_name) -> std::string {
            return apply_callback_rtc(kernel_name, precision);
        };
        queue.push({kernel_name, generate_src});
    }
}

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        puts("Usage: rocfft_aot_helper temp_cachefile.db output_cachefile.db "
             "path/to/rocfft_rtc_helper gfx000 gfx001 ...");
        return 1;
    }

    std::string              temp_cache_file   = argv[1];
    std::string              output_cache_file = argv[2];
    std::string              rtc_helper        = argv[3];
    std::vector<std::string> gpu_archs;
    for(int i = 4; i < argc; ++i)
        gpu_archs.push_back(argv[i]);

    // Default to using a persistent file in the current dir if no
    // cache file was given
    if(temp_cache_file.empty())
    {
        temp_cache_file = "rocfft_temp_cache.db";
    }

    // force RTC to use the temporary cache file
    rocfft_setenv("ROCFFT_RTC_CACHE_PATH", temp_cache_file.c_str());

    // disable system cache since we want to compile everything - use
    // an in-memory DB which will always be empty
    rocfft_setenv("ROCFFT_RTC_SYS_CACHE_PATH", ":memory:");

    // tell RTC where the compile helper is
    rocfft_setenv("ROCFFT_RTC_PROCESS_HELPER", rtc_helper.c_str());

    RTCCache::single = std::make_unique<RTCCache>();

    RTCCache::single->enable_write_mostly();

    CompileQueue queue;

    static const size_t      NUM_THREADS = rocfft_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for(size_t i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&queue, &gpu_archs]() {
            while(true)
            {
                auto item = queue.pop();
                if(item.kernel_name.empty())
                    break;
                for(const auto& gpu_arch : gpu_archs)
                    cached_compile(item.kernel_name, gpu_arch, item.generate_src, generator_sum());
            }
        });
    }

    build_stockham_function_pool(queue);
    build_realcomplex(queue);
    build_apply_callback(queue);

    // signal end of results with empty work items
    for(size_t i = 0; i < NUM_THREADS; ++i)
        queue.push({});
    for(size_t i = 0; i < NUM_THREADS; ++i)
        threads[i].join();

    // write the output file using what we collected in the temporary
    // cache
    RTCCache::single->write_aot_cache(output_cache_file, generator_sum(), gpu_archs);

    // try to shrink the temp cache file to 10 GiB
    try
    {
        RTCCache::single->cleanup_cache(static_cast<sqlite3_int64>(10) * 1024 * 1024 * 1024);
    }
    // the build should still succeed even if we fail to shrink the temp cache
    catch(std::exception&)
    {
        std::cerr << "warning: failed to shrink temp cache" << std::endl;
    }

    RTCCache::single.reset();

    return 0;
}
