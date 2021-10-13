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

#include <functional>

using namespace std::placeholders;

#include "../../shared/array_predicate.h"
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
#include "tree_node.h"

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
        kernel_name += "_twdbase" + std::to_string(node.largeTwdBase);

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
        else if(CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            kernel = std::make_unique<StockhamKernelRC>(specs);
        }
        else if(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
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
    src += real2complex_h;
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

    src += "static const bool apply_large_twiddle = ";
    if(node.large1D > 0)
        src += "true;\n";
    else
        src += "false;\n";

    src += "static const size_t large_twiddle_base = " + std::to_string(node.largeTwdBase) + ";\n";

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

std::vector<char> RTCKernel::compile(const std::string& kernel_src)
{
    hiprtcProgram prog;
    // give it a .cu extension so it'll be compiled as HIP code
    if(hiprtcCreateProgram(&prog, kernel_src.c_str(), "rocfft_rtc.cu", 0, nullptr, nullptr)
       != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }

    std::vector<const char*> options;
    options.push_back("-O3");
    options.push_back("-std=c++14");

    auto compileResult = hiprtcCompileProgram(prog, options.size(), options.data());
    if(compileResult != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(log.data());
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    hiprtcDestroyProgram(&prog);
    return code;
}

void RTCKernel::launch(DeviceCallIn& data)
{
    // arguments get pushed in an array of 64-bit values
    std::vector<void*> kargs;

    // twiddles
    kargs.push_back(data.node->twiddles.data());
    // large 1D twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        kargs.push_back(data.node->twiddles_large.data());
    // dim
    kargs.push_back(reinterpret_cast<void*>(data.node->length.size()));
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
                             gp.tpb_x,
                             gp.tpb_y,
                             gp.tpb_z,
                             gp.lds_bytes,
                             nullptr,
                             nullptr,
                             config)
       != hipSuccess)
        throw std::runtime_error("hipModuleLaunchKernel failure");
}

void RTCKernel::close_cache()
{
    // FIXME: reimplement
}

std::unique_ptr<RTCKernel>
    RTCKernel::runtime_compile(TreeNode& node, const char* gpu_arch, bool enable_callbacks)
{
#if ROCFFT_RUNTIME_COMPILE
    function_pool& pool = function_pool::get_function_pool();

    std::unique_ptr<StockhamGeneratorSpecs> specs;
    std::unique_ptr<StockhamGeneratorSpecs> specs2d;

    SBRC_TRANSPOSE_TYPE transpose_type = NONE;

    // SBRC variants look in the function pool for plain BLOCK_RC to
    // learn the block width, then decide on the transpose type once
    // that's known.
    auto pool_scheme = node.scheme;
    if(pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
       || pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
       || pool_scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
        pool_scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;

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
            return nullptr;

        // for SBRC variants, get the "real" kernel using the block
        // width and correct transpose type
        if(node.scheme != pool_scheme)
        {
            transpose_type = node.sbrc_3D_transpose_type(kernel.block_width);
            key            = fpkey(node.length[0], node.precision, node.scheme, transpose_type);
            kernel         = pool.get_kernel(key);
        }

        std::vector<unsigned int> factors;
        std::copy(kernel.factors.begin(), kernel.factors.end(), std::back_inserter(factors));

        specs = std::make_unique<StockhamGeneratorSpecs>(
            factors,
            std::vector<unsigned int>(),
            static_cast<unsigned int>(kernel.threads_per_block),
            PrintScheme(node.scheme));
        specs->threads_per_transform = kernel.threads_per_transform[0];
        specs->half_lds              = kernel.half_lds;
        specs->block_width           = kernel.block_width;
        break;
    }
    case CS_KERNEL_2D_SINGLE:
    {
        key              = fpkey(node.length[0], node.length[1], node.precision, node.scheme);
        FFTKernel kernel = pool.get_kernel(key);
        // already precompiled?
        if(kernel.device_function)
            return nullptr;

        std::vector<unsigned int> factors1d;
        std::vector<unsigned int> factors2d;

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
            static_cast<unsigned int>(kernel.threads_per_block),
            PrintScheme(node.scheme));
        specs->threads_per_transform = kernel.threads_per_transform[0];
        specs->half_lds              = kernel.half_lds;
        specs->block_width           = kernel.block_width;

        specs2d = std::make_unique<StockhamGeneratorSpecs>(
            factors2d,
            factors1d,
            static_cast<unsigned int>(kernel.threads_per_block),
            PrintScheme(node.scheme));
        specs2d->threads_per_transform = kernel.threads_per_transform[1];
        specs2d->half_lds              = kernel.half_lds;
        specs2d->block_width           = kernel.block_width;
        break;
    }
    default:
        return nullptr;
    }

    std::string kernel_name = stockham_rtc_kernel_name(node, transpose_type, enable_callbacks);

    // TODO: check the cache

    std::vector<char> code;

    auto kernel_src = stockham_rtc(
        *specs, specs2d ? *specs2d : *specs, kernel_name, node, transpose_type, enable_callbacks);

    if(LOG_RTC_ENABLED())
        (*LogSingleton::GetInstance().GetRTCOS()) << kernel_src << std::flush;

    // compile to code object
    code = compile(kernel_src);

    // TODO: store code object to cache

    return std::unique_ptr<RTCKernel>(new RTCKernel(kernel_name, code));
#else
    return nullptr;
#endif
}

rocfft_status rocfft_cache_serialize(void** buffer, size_t* buffer_len_bytes)
{
#if 0
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    auto& generator = get_generator();
    generator.open_db();
    PyObjWrap cache = PyRun_String("rtccache.serialize_cache(kernel_cache_db)",
                                   Py_eval_input,
                                   generator.glob_dict,
                                   generator.local_dict);
    if(cache.is_null())
    {
        PyErr_PrintEx(0);
        PyErr_Clear();
        return rocfft_status_failure;
    }

    size_t len        = PyBytes_Size(cache);
    *buffer_len_bytes = len;
    *buffer           = new char[len];
    memcpy(*buffer, PyBytes_AsString(cache), len);
    return rocfft_status_success;
#else
    return rocfft_status_failure;
#endif
}

rocfft_status rocfft_cache_buffer_free(void* buffer)
{
    delete[] static_cast<char*>(buffer);
    return rocfft_status_success;
}

rocfft_status rocfft_cache_deserialize(const void* buffer, size_t buffer_len_bytes)
{
#if 0
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    auto& generator = get_generator();
    generator.open_db();
    PyObjWrap cache
        = PyByteArray_FromStringAndSize(static_cast<const char*>(buffer), buffer_len_bytes);
    PyMapping_SetItemString(generator.local_dict, "cache", cache);
    PyObjWrap ret = PyRun_String("rtccache.deserialize_cache(kernel_cache_db, cache)",
                                 Py_eval_input,
                                 generator.glob_dict,
                                 generator.local_dict);
    if(ret.is_null())
    {
        PyErr_PrintEx(0);
        PyErr_Clear();
        return rocfft_status_failure;
    }

    return rocfft_status_success;
#else
    return rocfft_status_failure;
#endif
}
