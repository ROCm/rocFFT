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

#include <functional>

using namespace std::placeholders;

#include "../../shared/array_predicate.h"
#include "../../shared/environment.h"
#include "device/generator/generator.h"
#include "device/generator/stockham_gen.h"
#include "device/generator/stockham_gen_base.h"
#include "rtc.h"

#include "device/generator/stockham_gen_cc.h"
#include "device/generator/stockham_gen_cr.h"
#include "device/generator/stockham_gen_rc.h"
#include "device/generator/stockham_gen_rr.h"

#include "device/generator/stockham_gen_2d.h"

#include "device/kernel-generator-embed.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "logging.h"
#include "plan.h"
#include "rtccache.h"
#include "tree_node.h"

#include <chrono>

// generate name for RTC stockham kernel
//
// NOTE: this is the key for finding kernels in the cache, so distinct
// kernels *MUST* have unique names.
std::string stockham_rtc_kernel_name(const TreeNode&     node,
                                     SBRC_TRANSPOSE_TYPE transpose_type,
                                     bool                enable_callbacks)
{
    std::string kernel_name = "fft_rtc";

    if(node.direction == -1)
        kernel_name += "_fwd";
    else
        kernel_name += "_back";

    size_t length   = node.length.front();
    size_t length2D = node.scheme == CS_KERNEL_2D_SINGLE ? node.length[1] : 0;

    kernel_name += "_len";
    kernel_name += std::to_string(length);
    if(length2D)
        kernel_name += "x" + std::to_string(length2D);

    kernel_name += "_dim";
    kernel_name += std::to_string(node.length.size());

    auto array_type_name = [](rocfft_array_type type) {
        switch(type)
        {
        case rocfft_array_type_complex_interleaved:
            return "_CI";
        case rocfft_array_type_complex_planar:
            return "_CP";
        case rocfft_array_type_real:
            return "_R";
        case rocfft_array_type_hermitian_interleaved:
            return "_HI";
        case rocfft_array_type_hermitian_planar:
            return "_HP";
        default:
            return "_UN";
        }
    };

    kernel_name += node.precision == rocfft_precision_single ? "_sp" : "_dp";

    if(node.placement == rocfft_placement_inplace)
    {
        kernel_name += "_ip";
        kernel_name += array_type_name(node.inArrayType);
    }
    else
    {
        kernel_name += "_op";
        kernel_name += array_type_name(node.inArrayType);
        kernel_name += array_type_name(node.outArrayType);
    }

    if(node.inStride.front() == 1 && node.outStride.front() == 1)
        kernel_name += "_unitstride";

    switch(node.scheme)
    {
    case CS_KERNEL_STOCKHAM:
        kernel_name += "_sbrr";
        break;
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
        kernel_name += "_sbcc";
        break;
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
        kernel_name += "_sbcr";
        break;
    case CS_KERNEL_2D_SINGLE:
        // both lengths were already added above, which indicates it's
        // 2D_SINGLE
        break;
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    {
        kernel_name += "_sbrc";
        break;
    }
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    {
        auto transpose_type = kernel_name += "_sbrc_xy_z";
        break;
    }
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    {
        kernel_name += "_sbrc_z_xy";
        break;
    }
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
    {
        kernel_name += "_sbrc_erc_z_xy";
        break;
    }
    default:
        throw std::runtime_error("unsupported scheme in stockham_rtc_kernel_name");
    }

    switch(transpose_type)
    {
    case NONE:
        break;
    case DIAGONAL:
        kernel_name += "_diag";
        break;
    case TILE_ALIGNED:
        kernel_name += "_aligned";
        break;
    case TILE_UNALIGNED:
        kernel_name += "_unaligned";
        break;
    }

    if(node.large1D > 0)
    {
        kernel_name += "_twdbase" + std::to_string(node.largeTwdBase);
        kernel_name += "_" + std::to_string(node.ltwdSteps) + "step";
    }

    switch(node.ebtype)
    {
    case EmbeddedType::NONE:
        break;
    case EmbeddedType::C2Real_PRE:
        kernel_name += "_C2R";
        break;
    case EmbeddedType::Real2C_POST:
        kernel_name += "_R2C";
        break;
    }
    if(enable_callbacks)
        kernel_name += "_CB";
    return kernel_name;
}

std::string stockham_rtc(StockhamGeneratorSpecs& specs,
                         StockhamGeneratorSpecs& specs2d,
                         const std::string&      kernel_name,
                         TreeNode&               node,
                         SBRC_TRANSPOSE_TYPE     transpose_type,
                         bool                    enable_callbacks)
{
    std::unique_ptr<Function> device;
    std::unique_ptr<Function> device1;
    std::unique_ptr<Function> global;

    if(node.scheme == CS_KERNEL_2D_SINGLE)
    {
        StockhamKernelFused2D kernel(specs, specs2d);
        device = std::make_unique<Function>(kernel.kernel0.generate_device_function());
        if(kernel.kernel0.length != kernel.kernel1.length)
            device1 = std::make_unique<Function>(kernel.kernel1.generate_device_function());
        global = std::make_unique<Function>(kernel.generate_global_function());
    }
    else
    {
        std::unique_ptr<StockhamKernel> kernel;
        if(node.scheme == CS_KERNEL_STOCKHAM)
            kernel = std::make_unique<StockhamKernelRR>(specs);
        else if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            kernel = std::make_unique<StockhamKernelCC>(specs);
        else if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CR)
            kernel = std::make_unique<StockhamKernelCR>(specs);
        else if(node.scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            kernel = std::make_unique<StockhamKernelRC>(specs);
        }
        else if(node.scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(node.scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(node.scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else
            throw std::runtime_error("unhandled scheme");
        device = std::make_unique<Function>(kernel->generate_device_function());
        global = std::make_unique<Function>(kernel->generate_global_function());
    }

    // generated functions default to forward in-place interleaved.
    // adjust for direction, placement, format.
    if(node.direction == 1)
    {
        *device = make_inverse(*device);
        if(device1)
            *device1 = make_inverse(*device1);
        *global = make_inverse(*global);
    }
    if(node.placement == rocfft_placement_notinplace)
    {
        *device = make_outofplace(*device);
        if(device1)
            *device1 = make_outofplace(*device1);
        *global = make_outofplace(*global);
        if(array_type_is_planar(node.inArrayType))
            *global = make_planar(*global, "buf_in");
        if(array_type_is_planar(node.outArrayType))
            *global = make_planar(*global, "buf_out");
    }
    else
    {
        if(array_type_is_planar(node.inArrayType))
            *global = make_planar(*global, "buf");
    }

    // start off with includes
    std::string src = "// ROCFFT_RTC_BEGIN " + kernel_name + "\n";
    // callbacks are always potentially enabled, and activated by
    // checking the enable_callbacks variable later
    src += "#define ROCFFT_CALLBACKS_ENABLED\n";
    src += common_h;
    src += callback_h;
    src += butterfly_constant_h;
    src += rocfft_butterfly_template_h;
    src += real2complex_device_h;
    src += rtc_workarounds_h;

    src += device->render();
    if(device1)
        src += device1->render();

    // make_rtc removes templates from global function - add typedefs
    // and constants to replace them
    switch(node.precision)
    {
    case rocfft_precision_single:
        src += "typedef float2 scalar_type;\n";
        break;
    case rocfft_precision_double:
        src += "typedef double2 scalar_type;\n";
        break;
    }
    if(node.inStride.front() == 1 && node.outStride.front() == 1)
        src += "static const StrideBin sb = SB_UNIT;\n";
    else
        src += "static const StrideBin sb = SB_NONUNIT;\n";

    switch(node.ebtype)
    {
    case EmbeddedType::NONE:
        src += "static const EmbeddedType ebtype = EmbeddedType::NONE;\n";
        break;
    case EmbeddedType::Real2C_POST:
        src += "static const EmbeddedType ebtype = EmbeddedType::Real2C_POST;\n";
        break;
    case EmbeddedType::C2Real_PRE:
        src += "static const EmbeddedType ebtype = EmbeddedType::C2Real_PRE;\n";
        break;
    }

    // SBRC-specific template parameters that are ignored for other kernels
    switch(node.scheme)
    {
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        src += "static const SBRC_TYPE sbrc_type = SBRC_3D_FFT_TRANS_XY_Z;\n";
        break;
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        src += "static const SBRC_TYPE sbrc_type = SBRC_3D_FFT_TRANS_Z_XY;\n";
        break;
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
        src += "static const SBRC_TYPE sbrc_type = SBRC_3D_FFT_ERC_TRANS_Z_XY;\n";
        break;
    default:
        src += "static const SBRC_TYPE sbrc_type = SBRC_2D;\n";
    }
    switch(transpose_type)
    {
    case NONE:
        src += "static const SBRC_TRANSPOSE_TYPE transpose_type = NONE;\n";
        break;
    case DIAGONAL:
        src += "static const SBRC_TRANSPOSE_TYPE transpose_type = DIAGONAL;\n";
        break;
    case TILE_ALIGNED:
        src += "static const SBRC_TRANSPOSE_TYPE transpose_type = TILE_ALIGNED;\n";
        break;
    case TILE_UNALIGNED:
        src += "static const SBRC_TRANSPOSE_TYPE transpose_type = TILE_UNALIGNED;\n";
        break;
    }

    if(enable_callbacks)
        src += "static const CallbackType cbtype = CallbackType::USER_LOAD_STORE;\n";
    else
        src += "static const CallbackType cbtype = CallbackType::NONE;\n";

    switch(node.dir2regMode)
    {
    case DirectRegType::FORCE_OFF_OR_NOT_SUPPORT:
        src += "static const DirectRegType drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;\n";
        break;
    case DirectRegType::TRY_ENABLE_IF_SUPPORT:
        src += "static const DirectRegType drtype = DirectRegType::TRY_ENABLE_IF_SUPPORT;\n";
        break;
    }

    src += "static const bool apply_large_twiddle = ";
    if(node.large1D > 0)
        src += "true;\n";
    else
        src += "false;\n";

    src += "static const size_t large_twiddle_base = " + std::to_string(node.largeTwdBase) + ";\n";
    src += "static const size_t large_twiddle_steps = " + std::to_string(node.ltwdSteps) + ";\n";

    src += make_rtc(*global, kernel_name).render();
    src += "// ROCFFT_RTC_END " + kernel_name + "\n";
    return src;
}

RTCKernel::RTCKernel(const std::string& kernel_name, const std::vector<char>& code)
{
    if(hipModuleLoadData(&module, code.data()) != hipSuccess)
        throw std::runtime_error("failed to load module");

    if(hipModuleGetFunction(&kernel, module, kernel_name.c_str()) != hipSuccess)
        throw std::runtime_error("failed to get function");
}

void RTCKernel::launch(DeviceCallIn& data)
{
    // arguments get pushed in an array of 64-bit values
    std::vector<void*> kargs;

    // twiddles
    kargs.push_back(data.node->twiddles);
    // large 1D twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        kargs.push_back(data.node->twiddles_large);
    // lengths
    kargs.push_back(kargs_lengths(data.node->devKernArg));
    // stride in/out
    kargs.push_back(kargs_stride_in(data.node->devKernArg));
    if(data.node->placement == rocfft_placement_notinplace)
        kargs.push_back(kargs_stride_out(data.node->devKernArg));
    // nbatch
    kargs.push_back(reinterpret_cast<void*>(data.node->batch));
    // lds padding
    kargs.push_back(reinterpret_cast<void*>(data.node->lds_padding));
    // callback params
    kargs.push_back(data.callbacks.load_cb_fn);
    kargs.push_back(data.callbacks.load_cb_data);
    kargs.push_back(reinterpret_cast<void*>(data.callbacks.load_cb_lds_bytes));
    kargs.push_back(data.callbacks.store_cb_fn);
    kargs.push_back(data.callbacks.store_cb_data);

    // buffer pointers
    kargs.push_back(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.push_back(data.bufIn[1]);
    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.push_back(data.bufOut[0]);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.push_back(data.bufOut[1]);
    }

    auto  size     = sizeof(kargs.size() * sizeof(void*));
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      kargs.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      HIP_LAUNCH_PARAM_END};

    const auto& gp = data.gridParam;

    if(hipModuleLaunchKernel(kernel,
                             gp.b_x,
                             gp.b_y,
                             gp.b_z,
                             gp.wgs_x,
                             gp.wgs_y,
                             gp.wgs_z,
                             gp.lds_bytes,
                             data.rocfft_stream,
                             nullptr,
                             config)
       != hipSuccess)
        throw std::runtime_error("hipModuleLaunchKernel failure");
}

// allow user control of whether RTC is done in-process or out-of-process
enum class RTCProcessType
{
    // allow one in-process compile, fallback to out-of-process if
    // one is already in progress.  fall back further to waiting for
    // the lock if subprocess failed.
    DEFAULT,
    // only try in-process, waiting for lock if necessary
    FORCE_IN_PROCESS,
    // only try out-of-process, never needs lock
    FORCE_OUT_PROCESS,
};

static RTCProcessType get_rtc_process_type()
{
    auto var = rocfft_getenv("ROCFFT_RTC_PROCESS");
    // defined and equal to 0 means force in-process
    if(var == "0")
        return RTCProcessType::FORCE_IN_PROCESS;
    // defined and equal to 1 means force out-process
    if(var == "1")
        return RTCProcessType::FORCE_OUT_PROCESS;
    // ideal default behaviour - try in-process first and use
    // out-process if necessary
    if(var == "2")
        return RTCProcessType::DEFAULT;

    return RTCProcessType::DEFAULT;
}

std::shared_future<std::unique_ptr<RTCKernel>>
    RTCKernel::runtime_compile(TreeNode& node, const std::string& gpu_arch, bool enable_callbacks)
{
#ifdef ROCFFT_RUNTIME_COMPILE
    function_pool& pool = function_pool::get_function_pool();

    std::unique_ptr<StockhamGeneratorSpecs> specs;
    std::unique_ptr<StockhamGeneratorSpecs> specs2d;

    SBRC_TRANSPOSE_TYPE transpose_type = NONE;

    // SBRC variants look in the function pool for plain BLOCK_RC to
    // learn the block width, then decide on the transpose type once
    // that's known.
    auto         pool_scheme = node.scheme;
    unsigned int static_dim  = node.length.size();
    if(pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
       || pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
       || pool_scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
    {
        pool_scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;
        // These are all 3D kernels, but are sometimes shoehorned
        // into 2D plans.  Make sure they get at least 3 dims.
        if(static_dim == 2)
            static_dim = 3;
    }

    // find function pool entry so we can construct specs for the generator
    FMKey key;
    switch(pool_scheme)
    {
    case CS_KERNEL_STOCKHAM:
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    {
        // these go into the function pool normally and are passed to
        // the generator as-is
        key              = fpkey(node.length[0], node.precision, pool_scheme);
        FFTKernel kernel = pool.get_kernel(key);
        // already precompiled?
        if(kernel.device_function)
        {
            std::promise<std::unique_ptr<RTCKernel>> p;
            p.set_value(nullptr);
            return p.get_future();
        }

        // for SBRC variants, get the "real" kernel using the block
        // width and correct transpose type
        if(pool_scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            transpose_type = node.sbrc_transpose_type(kernel.transforms_per_block);
            key            = fpkey(node.length[0], node.precision, node.scheme, transpose_type);
            kernel         = pool.get_kernel(key);
        }

        std::vector<unsigned int> factors;
        std::copy(kernel.factors.begin(), kernel.factors.end(), std::back_inserter(factors));
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        specs = std::make_unique<StockhamGeneratorSpecs>(
            factors,
            std::vector<unsigned int>(),
            precisions,
            static_cast<unsigned int>(kernel.workgroup_size),
            PrintScheme(node.scheme));
        specs->threads_per_transform = kernel.threads_per_transform[0];
        specs->half_lds              = kernel.half_lds;
        specs->direct_to_from_reg    = kernel.direct_to_from_reg;
        specs->static_dim            = static_dim;
        break;
    }
    case CS_KERNEL_2D_SINGLE:
    {
        key              = fpkey(node.length[0], node.length[1], node.precision, node.scheme);
        FFTKernel kernel = pool.get_kernel(key);
        // already precompiled?
        if(kernel.device_function)
        {
            std::promise<std::unique_ptr<RTCKernel>> p;
            p.set_value(nullptr);
            return p.get_future();
        }

        std::vector<unsigned int> factors1d;
        std::vector<unsigned int> factors2d;
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        // need to break down factors into first dim and second dim
        size_t len0_remain = node.length[0];
        for(auto& f : kernel.factors)
        {
            len0_remain /= f;
            if(len0_remain > 0)
            {
                factors1d.push_back(f);
            }
            else
            {
                factors2d.push_back(f);
            }
        }

        specs = std::make_unique<StockhamGeneratorSpecs>(
            factors1d,
            factors2d,
            precisions,
            static_cast<unsigned int>(kernel.workgroup_size),
            PrintScheme(node.scheme));
        specs->threads_per_transform = kernel.threads_per_transform[0];
        specs->half_lds              = kernel.half_lds;
        specs->static_dim            = static_dim;

        specs2d = std::make_unique<StockhamGeneratorSpecs>(
            factors2d,
            factors1d,
            precisions,
            static_cast<unsigned int>(kernel.workgroup_size),
            PrintScheme(node.scheme));
        specs2d->threads_per_transform = kernel.threads_per_transform[1];
        specs2d->half_lds              = kernel.half_lds;
        break;
    }
    default:
    {
        std::promise<std::unique_ptr<RTCKernel>> p;
        p.set_value(nullptr);
        return p.get_future();
    }
    }

    std::string kernel_name = stockham_rtc_kernel_name(node, transpose_type, enable_callbacks);

    // check the cache
    std::vector<char> code;

    int hip_version = 0;
    if(hipRuntimeGetVersion(&hip_version) != hipSuccess)
    {
        std::promise<std::unique_ptr<RTCKernel>> p;
        p.set_value(nullptr);
        return p.get_future();
    }

    std::vector<char> generator_sum_vec(generator_sum, generator_sum + generator_sum_bytes);
    if(RTCCache::single)
    {
        code = RTCCache::single->get_code_object(
            kernel_name, gpu_arch, hip_version, generator_sum_vec);
    }

    if(!code.empty())
    {
        // cache hit
        try
        {
            if(LOG_RTC_ENABLED())
            {
                (*LogSingleton::GetInstance().GetRTCOS())
                    << "// cache hit for " << kernel_name << std::endl;
            }
            std::promise<std::unique_ptr<RTCKernel>> p;
            p.set_value(std::unique_ptr<RTCKernel>(new RTCKernel(kernel_name, code)));
            return p.get_future();
        }
        catch(std::exception&)
        {
            // if for some reason the cached object was not
            // usable, fall through to generating the source and
            // recompiling
            if(LOG_RTC_ENABLED())
            {
                (*LogSingleton::GetInstance().GetRTCOS())
                    << "// cache unusable for " << kernel_name << std::endl;
            }
        }
    }

    // otherwise, we did not find a cached code object, and need to
    // compile the source

    // compile to code object
    return std::async(
        std::launch::async, [=, &node, specs = move(specs), specs2d = move(specs2d)]() {
            auto generate_begin = std::chrono::steady_clock::now();
            auto kernel_src     = stockham_rtc(*specs,
                                           specs2d ? *specs2d : *specs,
                                           kernel_name,
                                           node,
                                           transpose_type,
                                           enable_callbacks);
            auto generate_end   = std::chrono::steady_clock::now();

            if(LOG_RTC_ENABLED())
            {
                std::chrono::duration<float, std::milli> generate_ms
                    = generate_end - generate_begin;

                (*LogSingleton::GetInstance().GetRTCOS())
                    << kernel_src << "// " << kernel_name
                    << " generate duration: " << static_cast<int>(generate_ms.count()) << " ms"
                    << std::endl;
            }

            std::vector<char> code;
            // try to set compile_begin time right when we're really
            // about to compile (i.e. after acquiring any locks)
            std::chrono::time_point<std::chrono::steady_clock> compile_begin;

            RTCProcessType process_type = get_rtc_process_type();
            switch(process_type)
            {
            case RTCProcessType::FORCE_OUT_PROCESS:
            {
                compile_begin = std::chrono::steady_clock::now();
                try
                {
                    code = compile_subprocess(kernel_src);
                    break;
                }
                catch(std::exception&)
                {
                    // if subprocess had a problem, ignore it and
                    // fall through to forced-in-process compile
                }
            }
            case RTCProcessType::FORCE_IN_PROCESS:
            {
                std::lock_guard<std::mutex> lck(compile_lock);
                compile_begin = std::chrono::steady_clock::now();
                code          = compile(kernel_src);
                break;
            }
            default:
            {
                // do it in-process if possible
                std::unique_lock<std::mutex> lock(compile_lock, std::try_to_lock);
                if(lock.owns_lock())
                {
                    compile_begin = std::chrono::steady_clock::now();
                    code          = compile(kernel_src);
                    lock.unlock();
                }
                else
                {
                    // couldn't acquire lock, so try instead in a subprocess
                    try
                    {
                        compile_begin = std::chrono::steady_clock::now();
                        code          = compile_subprocess(kernel_src);
                    }
                    catch(std::exception&)
                    {
                        // subprocess still didn't work, re-acquire lock
                        // and fall back to in-process if something went
                        // wrong
                        std::lock_guard<std::mutex> lck(compile_lock);
                        compile_begin = std::chrono::steady_clock::now();
                        code          = compile(kernel_src);
                    }
                }
            }
            }
            auto compile_end = std::chrono::steady_clock::now();

            if(LOG_RTC_ENABLED())
            {
                std::chrono::duration<float, std::milli> compile_ms = compile_end - compile_begin;

                (*LogSingleton::GetInstance().GetRTCOS())
                    << "// " << kernel_name
                    << " compile duration: " << static_cast<int>(compile_ms.count()) << " ms\n"
                    << std::endl;
            }

            if(RTCCache::single)
            {
                RTCCache::single->store_code_object(
                    kernel_name, gpu_arch, hip_version, generator_sum_vec, code);
            }
            return std::unique_ptr<RTCKernel>(new RTCKernel(kernel_name, code));
        });
#else
    // runtime compilation is not enabled, return null RTCKernel
    std::promise<std::unique_ptr<RTCKernel>> p;
    p.set_value(nullptr);
    return p.get_future();
#endif
}

rocfft_status rocfft_cache_serialize(void** buffer, size_t* buffer_len_bytes)
{
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    if(!RTCCache::single)
        return rocfft_status_failure;

    return RTCCache::single->serialize(buffer, buffer_len_bytes);
}

rocfft_status rocfft_cache_buffer_free(void* buffer)
{
    if(!RTCCache::single)
        return rocfft_status_failure;
    RTCCache::single->serialize_free(buffer);
    return rocfft_status_success;
}

rocfft_status rocfft_cache_deserialize(const void* buffer, size_t buffer_len_bytes)
{
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    if(!RTCCache::single)
        return rocfft_status_failure;

    return RTCCache::single->deserialize(buffer, buffer_len_bytes);
}
