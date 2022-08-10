// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/array_predicate.h"
#include "rtc_stockham.h"

using namespace std::placeholders;

#include "device/generator/generator.h"
#include "device/generator/stockham_gen.h"
#include "device/generator/stockham_gen_base.h"

#include "device/generator/stockham_gen_cc.h"
#include "device/generator/stockham_gen_cr.h"
#include "device/generator/stockham_gen_rc.h"
#include "device/generator/stockham_gen_rr.h"

#include "device/generator/stockham_gen_2d.h"

#include "device/kernel-generator-embed.h"

#include "function_pool.h"
#include "kernel_launch.h"
#include "tree_node.h"

// generate name for RTC stockham kernel
std::string stockham_rtc_kernel_name(ComputeScheme           scheme,
                                     size_t                  length1D,
                                     size_t                  length2D,
                                     size_t                  static_dim,
                                     int                     direction,
                                     rocfft_precision        precision,
                                     rocfft_result_placement placement,
                                     rocfft_array_type       inArrayType,
                                     rocfft_array_type       outArrayType,
                                     bool                    unitstride,
                                     size_t                  largeTwdBase,
                                     size_t                  largeTwdSteps,
                                     EmbeddedType            ebtype,
                                     DirectRegType           dir2regMode,
                                     IntrinsicAccessType     intrinsicMode,
                                     SBRC_TRANSPOSE_TYPE     transpose_type,
                                     bool                    enable_callbacks,
                                     bool                    enable_scaling)
{
    std::string kernel_name = "fft_rtc";

    if(direction == -1)
        kernel_name += "_fwd";
    else
        kernel_name += "_back";

    kernel_name += "_len";
    kernel_name += std::to_string(length1D);
    if(length2D)
        kernel_name += "x" + std::to_string(length2D);

    if(static_dim)
    {
        kernel_name += "_dim";
        kernel_name += std::to_string(static_dim);
    }

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

    kernel_name += precision == rocfft_precision_single ? "_sp" : "_dp";

    if(placement == rocfft_placement_inplace)
    {
        kernel_name += "_ip";
        kernel_name += array_type_name(inArrayType);
    }
    else
    {
        kernel_name += "_op";
        kernel_name += array_type_name(inArrayType);
        kernel_name += array_type_name(outArrayType);
    }

    if(unitstride)
        kernel_name += "_unitstride";

    switch(scheme)
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

    if(largeTwdBase > 0 && largeTwdSteps > 0)
    {
        kernel_name += "_twdbase" + std::to_string(largeTwdBase);
        kernel_name += "_" + std::to_string(largeTwdSteps) + "step";
    }

    switch(ebtype)
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

    if(dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)
        kernel_name += "_dirReg";

    switch(intrinsicMode)
    {
    case IntrinsicAccessType::DISABLE_BOTH:
        break;
    case IntrinsicAccessType::ENABLE_BOTH:
        kernel_name += "_intrinsicReadWrite";
        break;
    case IntrinsicAccessType::ENABLE_LOAD_ONLY:
        kernel_name += "_intrinsicRead";
        break;
    }

    if(enable_callbacks)
        kernel_name += "_CB";
    if(enable_scaling)
        kernel_name += "_scale";
    return kernel_name;
}

std::string stockham_rtc(const StockhamGeneratorSpecs& specs,
                         const StockhamGeneratorSpecs& specs2d,
                         unsigned int*                 transforms_per_block,
                         const std::string&            kernel_name,
                         ComputeScheme                 scheme,
                         int                           direction,
                         rocfft_precision              precision,
                         rocfft_result_placement       placement,
                         rocfft_array_type             inArrayType,
                         rocfft_array_type             outArrayType,
                         bool                          unit_stride,
                         size_t                        largeTwdBase,
                         size_t                        largeTwdSteps,
                         EmbeddedType                  ebtype,
                         DirectRegType                 dir2regMode,
                         IntrinsicAccessType           intrinsicMode,
                         SBRC_TRANSPOSE_TYPE           transpose_type,
                         bool                          enable_callbacks,
                         bool                          enable_scaling)
{
    std::unique_ptr<Function> lds2reg, reg2lds, device;
    std::unique_ptr<Function> lds2reg1, reg2lds1, device1;
    std::unique_ptr<Function> global;

    if(scheme == CS_KERNEL_2D_SINGLE)
    {
        StockhamKernelFused2D kernel(specs, specs2d);
        if(transforms_per_block)
            *transforms_per_block = kernel.transforms_per_block;
        lds2reg = std::make_unique<Function>(kernel.kernel0.generate_lds_to_reg_input_function());
        reg2lds
            = std::make_unique<Function>(kernel.kernel0.generate_lds_from_reg_output_function());
        device = std::make_unique<Function>(kernel.kernel0.generate_device_function());
        if(kernel.kernel0.length != kernel.kernel1.length)
        {
            lds2reg1
                = std::make_unique<Function>(kernel.kernel1.generate_lds_to_reg_input_function());
            reg2lds1 = std::make_unique<Function>(
                kernel.kernel1.generate_lds_from_reg_output_function());
            device1 = std::make_unique<Function>(kernel.kernel1.generate_device_function());
        }
        global = std::make_unique<Function>(kernel.generate_global_function());
    }
    else
    {
        std::unique_ptr<StockhamKernel> kernel;
        if(scheme == CS_KERNEL_STOCKHAM)
            kernel = std::make_unique<StockhamKernelRR>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            kernel = std::make_unique<StockhamKernelCC>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CR)
            kernel = std::make_unique<StockhamKernelCR>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            kernel = std::make_unique<StockhamKernelRC>(specs);
        }
        else if(scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
            kernel = std::make_unique<StockhamKernelRC>(specs);
        else
            throw std::runtime_error("unhandled scheme");
        if(transforms_per_block)
            *transforms_per_block = kernel->transforms_per_block;
        lds2reg = std::make_unique<Function>(kernel->generate_lds_to_reg_input_function());
        reg2lds = std::make_unique<Function>(kernel->generate_lds_from_reg_output_function());
        device  = std::make_unique<Function>(kernel->generate_device_function());
        global  = std::make_unique<Function>(kernel->generate_global_function());
    }

    // generated functions default to forward in-place interleaved.
    // adjust for direction, placement, format.
    if(direction == 1)
    {
        *device = make_inverse(*device);
        if(device1)
            *device1 = make_inverse(*device1);
        *global = make_inverse(*global);
    }
    if(placement == rocfft_placement_notinplace)
    {
        *global = make_outofplace(*global);
        if(array_type_is_planar(inArrayType))
            *global = make_planar(*global, "buf_in");
        if(array_type_is_planar(outArrayType))
            *global = make_planar(*global, "buf_out");
    }
    else
    {
        if(array_type_is_planar(inArrayType))
            *global = make_planar(*global, "buf");
    }

    // start off with includes
    std::string src = "// ROCFFT_RTC_BEGIN " + kernel_name + "\n";
    // callbacks are always potentially enabled, and activated by
    // checking the enable_callbacks variable later
    src += "#define ROCFFT_CALLBACKS_ENABLED\n";
    src += common_h;
    src += memory_gfx_h;
    src += callback_h;
    src += butterfly_constant_h;
    src += rocfft_butterfly_template_h;
    src += real2complex_device_h;
    src += rtc_workarounds_h;

    src += lds2reg->render();
    src += reg2lds->render();
    src += device->render();
    if(lds2reg1)
        src += lds2reg1->render();
    if(reg2lds1)
        src += reg2lds1->render();
    if(device1)
        src += device1->render();

    // make_rtc removes templates from global function - add typedefs
    // and constants to replace them
    switch(precision)
    {
    case rocfft_precision_single:
        src += "typedef float2 scalar_type;\n";
        break;
    case rocfft_precision_double:
        src += "typedef double2 scalar_type;\n";
        break;
    }
    if(unit_stride)
        src += "static const StrideBin sb = SB_UNIT;\n";
    else
        src += "static const StrideBin sb = SB_NONUNIT;\n";

    switch(ebtype)
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
    switch(scheme)
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

    switch(dir2regMode)
    {
    case DirectRegType::FORCE_OFF_OR_NOT_SUPPORT:
        src += "static const DirectRegType drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;\n";
        break;
    case DirectRegType::TRY_ENABLE_IF_SUPPORT:
        src += "static const DirectRegType drtype = DirectRegType::TRY_ENABLE_IF_SUPPORT;\n";
        break;
    }

    src += "static const bool apply_large_twiddle = ";
    src += (largeTwdBase > 0 && largeTwdSteps > 0) ? "true;\n" : "false;\n";

    switch(intrinsicMode)
    {
    case IntrinsicAccessType::DISABLE_BOTH:
        src += "static const IntrinsicAccessType intrinsic_mode = "
               "IntrinsicAccessType::DISABLE_BOTH;\n";
        break;
    case IntrinsicAccessType::ENABLE_BOTH:
        src += "static const IntrinsicAccessType intrinsic_mode = "
               "IntrinsicAccessType::ENABLE_BOTH;\n";
        break;
    case IntrinsicAccessType::ENABLE_LOAD_ONLY:
        src += "static const IntrinsicAccessType intrinsic_mode = "
               "IntrinsicAccessType::ENABLE_LOAD_ONLY;\n";
        break;
    }

    src += "static const size_t large_twiddle_base = " + std::to_string(largeTwdBase) + ";\n";
    src += "static const size_t large_twiddle_steps = " + std::to_string(largeTwdSteps) + ";\n";

    src += make_rtc(*global, kernel_name, enable_scaling).render();
    src += "// ROCFFT_RTC_END " + kernel_name + "\n";
    return src;
}

RTCKernel::RTCGenerator RTCKernelStockham::generate_from_node(const TreeNode&    node,
                                                              const std::string& gpu_arch,
                                                              bool               enable_callbacks)
{
    RTCGenerator   generator;
    function_pool& pool = function_pool::get_function_pool();

    std::optional<StockhamGeneratorSpecs> specs;
    std::optional<StockhamGeneratorSpecs> specs2d;

    // if scale factor is enabled, we force RTC for this kernel
    bool enable_scaling = node.IsScalingEnabled();

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

    std::optional<FFTKernel> kernel;

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
        key    = fpkey(node.length[0], node.precision, pool_scheme);
        kernel = pool.get_kernel(key);
        // already precompiled?
        if(kernel->device_function && !enable_scaling)
        {
            return generator;
        }

        // for SBRC variants, get the "real" kernel using the block
        // width and correct transpose type
        if(pool_scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            transpose_type = node.sbrc_transpose_type(kernel->transforms_per_block);
            key            = fpkey(node.length[0], node.precision, node.scheme, transpose_type);
            kernel         = pool.get_kernel(key);
        }

        std::vector<unsigned int> factors;
        std::copy(kernel->factors.begin(), kernel->factors.end(), std::back_inserter(factors));
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        specs.emplace(factors,
                      std::vector<unsigned int>(),
                      precisions,
                      static_cast<unsigned int>(kernel->workgroup_size),
                      PrintScheme(node.scheme));
        specs->threads_per_transform = kernel->threads_per_transform[0];
        specs->half_lds              = kernel->half_lds;
        specs->direct_to_from_reg    = kernel->direct_to_from_reg;
        break;
    }
    case CS_KERNEL_2D_SINGLE:
    {
        key    = fpkey(node.length[0], node.length[1], node.precision, node.scheme);
        kernel = pool.get_kernel(key);
        // already precompiled?
        if(kernel->device_function && !enable_scaling)
        {
            return generator;
        }

        std::vector<unsigned int> factors1d;
        std::vector<unsigned int> factors2d;
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        // need to break down factors into first dim and second dim
        size_t len0_remain = node.length[0];
        for(auto& f : kernel->factors)
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

        specs.emplace(factors1d,
                      factors2d,
                      precisions,
                      static_cast<unsigned int>(kernel->workgroup_size),
                      PrintScheme(node.scheme));
        specs->threads_per_transform = kernel->threads_per_transform[0];
        specs->half_lds              = kernel->half_lds;

        specs2d.emplace(factors2d,
                        factors1d,
                        precisions,
                        static_cast<unsigned int>(kernel->workgroup_size),
                        PrintScheme(node.scheme));
        specs2d->threads_per_transform = kernel->threads_per_transform[1];
        specs2d->half_lds              = kernel->half_lds;
        break;
    }
    default:
    {
        return generator;
    }
    }

    // static_dim has what the plan requires, and RTC kernels are
    // built for exactly that dimension.  But kernels that are
    // compiled ahead of time are general across all dims and take
    // 'dim' as an argument.  So set static_dim to 0 to communicate
    // this to the generator and launch machinery.
    if(kernel && kernel->aot_rtc)
        static_dim = 0;
    specs->static_dim = static_dim;

    bool unit_stride = node.inStride.front() == 1 && node.outStride.front() == 1;

    generator.generate_name = [=, &node]() {
        return stockham_rtc_kernel_name(node.scheme,
                                        node.length[0],
                                        node.scheme == CS_KERNEL_2D_SINGLE ? node.length[1] : 0,
                                        static_dim,
                                        node.direction,
                                        node.precision,
                                        node.placement,
                                        node.inArrayType,
                                        node.outArrayType,
                                        unit_stride,
                                        node.largeTwdBase,
                                        node.ltwdSteps,
                                        node.ebtype,
                                        node.dir2regMode,
                                        node.intrinsicMode,
                                        transpose_type,
                                        enable_callbacks,
                                        node.IsScalingEnabled());
    };

    generator.generate_src = [=, &node](const std::string& kernel_name) {
        return stockham_rtc(*specs,
                            specs2d ? *specs2d : *specs,
                            nullptr,
                            kernel_name,
                            node.scheme,
                            node.direction,
                            node.precision,
                            node.placement,
                            node.inArrayType,
                            node.outArrayType,
                            unit_stride,
                            node.largeTwdBase,
                            node.ltwdSteps,
                            node.ebtype,
                            node.dir2regMode,
                            node.intrinsicMode,
                            transpose_type,
                            enable_callbacks,
                            node.IsScalingEnabled());
    };

    generator.construct_rtckernel
        = [](const std::string& kernel_name, const std::vector<char>& code) {
              return std::unique_ptr<RTCKernel>(new RTCKernelStockham(kernel_name, code));
          };
    return generator;
}

RTCKernelArgs RTCKernelStockham::get_launch_args(DeviceCallIn& data)
{
    // construct arguments to pass to the kernel
    RTCKernelArgs kargs;

    // twiddles
    kargs.append_ptr(data.node->twiddles);
    // large 1D twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        kargs.append_ptr(data.node->twiddles_large);
    if(!hardcoded_dim)
        kargs.append_size_t(data.node->length.size());
    // lengths
    kargs.append_ptr(kargs_lengths(data.node->devKernArg));
    // stride in/out
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    if(data.node->placement == rocfft_placement_notinplace)
        kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
    // nbatch
    kargs.append_size_t(data.node->batch);
    // lds padding
    kargs.append_unsigned_int(data.node->lds_padding);
    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);

    // buffer pointers
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.append_ptr(data.bufOut[0]);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.append_ptr(data.bufOut[1]);
    }
    return kargs;
}
