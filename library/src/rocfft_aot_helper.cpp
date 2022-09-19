#include <functional>
#include <thread>

using namespace std::placeholders;

#include "../../shared/environment.h"
#include "function_pool.h"
#include "rtc_cache.h"
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

#include <condition_variable>
#include <mutex>
#include <queue>
struct CompileQueue
{
    struct WorkItem
    {
        std::string      kernel_name;
        kernel_src_gen_t generate_src;
    };
    void push(WorkItem&& i)
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        items.emplace(std::move(i));
        emptyWait.notify_all();
    }
    WorkItem pop()
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        while(items.empty())
            emptyWait.wait(lock);
        WorkItem item(items.front());
        items.pop();
        return item;
    }

private:
    std::queue<WorkItem>    items;
    std::mutex              queueMutex;
    std::condition_variable emptyWait;
};

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
                                       size_t,
                                       size_t,
                                       bool,
                                       bool)> func)
{
    std::vector<bool>                    unitstride_range = {false};
    std::vector<rocfft_result_placement> placements       = {rocfft_placement_notinplace};
    std::vector<EmbeddedType>            ebtypes          = {EmbeddedType::NONE};
    std::vector<SBRC_TRANSPOSE_TYPE>     sbrc_trans_types = {SBRC_TRANSPOSE_TYPE::NONE};
    std::vector<DirectRegType>           dir_reg_types    = {FORCE_OFF_OR_NOT_SUPPORT};
    std::vector<IntrinsicAccessType>     intrinsic_modes  = {DISABLE_BOTH};
    std::vector<std::array<size_t, 2>>   base_steps
        = {{0, 0}, {4, 3}, {5, 3}, {6, 3}, {8, 2}, {8, 3}};

    switch(scheme)
    {
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    {
        // SBCC can be used with or without large twd.  Large
        // twd may be base 4, 5, 6, 8.  Base 8 can
        // be 2 or 3 steps; other bases are always 3 step.
        placements.push_back(rocfft_placement_inplace);
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    {
        base_steps.resize(1);
        unitstride_range.push_back(true);
        ebtypes.push_back(EmbeddedType::C2Real_PRE);
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
    {
        base_steps.resize(1);
        unitstride_range.push_back(true);
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
    default:
        throw std::runtime_error("unsupported scheme in stockham_combo aot_rtc");
    }

    // if no dir-to-reg support, then we don't have intrinsic buffer RW,
    // and only force_off_or_not_support for dir2reg. Else, we have all possibilities
    // (even force_off_or_not_support is still included)
    if(kernel.direct_to_from_reg)
    {
        dir_reg_types.push_back(TRY_ENABLE_IF_SUPPORT);
        intrinsic_modes.push_back(ENABLE_BOTH);
        intrinsic_modes.push_back(ENABLE_LOAD_ONLY);
    }

    for(auto direction : {-1, 1})
    {
        for(auto placement : placements)
        {
            for(auto inArrayType :
                {rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar})
            {
                for(auto outArrayType :
                    {rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar})
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

    static const std::set<ComputeScheme> aot_rtc_supported
        = {CS_KERNEL_STOCKHAM_BLOCK_CC,
           CS_KERNEL_STOCKHAM_BLOCK_CR,
           CS_KERNEL_STOCKHAM_BLOCK_RC,
           CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
           CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
           CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY};

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

        if(aot_rtc_supported.count(scheme))
        {
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
                                       size_t                  ltwd_base,
                                       size_t                  ltwd_step,
                                       bool                    unitstride,
                                       bool                    callbacks) {
                               // intrinsic mode require non-callback and enable dir_reg
                               if((callbacks || dir_reg_type == FORCE_OFF_OR_NOT_SUPPORT)
                                  && (intrinsic != IntrinsicAccessType::DISABLE_BOTH))
                                   return;

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
}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        puts("Usage: rocfft_aot_helper cachefile.db path/to/rocfft_rtc_helper gfx000 gfx001 ...");
        return 1;
    }

    std::string              cache_file = argv[1];
    std::string              rtc_helper = argv[2];
    std::vector<std::string> gpu_archs;
    for(int i = 3; i < argc; ++i)
        gpu_archs.push_back(argv[i]);

    // force RTC to use a clean temporary cache file
    auto temp_cache_file = fs::temp_directory_path() / "rocfft_temp_cache.db";
    if(fs::exists(temp_cache_file))
        fs::remove(temp_cache_file);
    rocfft_setenv("ROCFFT_RTC_CACHE_PATH", temp_cache_file.string().c_str());

    // disable system cache since we want to compile everything - use
    // an in-memory DB which will always be empty
    rocfft_setenv("ROCFFT_RTC_SYS_CACHE_PATH", ":memory:");

    // tell RTC where the compile helper is
    rocfft_setenv("ROCFFT_RTC_PROCESS_HELPER", rtc_helper.c_str());

    // force RTC to use subprocess, to work around asserts when combining multithreading + hipRTC
    rocfft_setenv("ROCFFT_RTC_PROCESS", "1");

    RTCCache::single = std::make_unique<RTCCache>();

    CompileQueue queue;

    static const size_t      NUM_THREADS = std::thread::hardware_concurrency();
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

    // signal end of results with empty work items
    for(size_t i = 0; i < NUM_THREADS; ++i)
        queue.push({});
    for(size_t i = 0; i < NUM_THREADS; ++i)
        threads[i].join();

    // write the output file using what we collected in the temporary
    // cache
    RTCCache::single->write_aot_cache(cache_file);

    RTCCache::single.reset();

    // clean up the temp file
    fs::remove(temp_cache_file);

    return 0;
}
