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
#include "rtc_stockham_gen.h"

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
                                     bool                    largeTwdBatchIsTransformCount,
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

    kernel_name += rtc_precision_name(precision);

    if(placement == rocfft_placement_inplace)
    {
        kernel_name += "_ip";
        kernel_name += rtc_array_type_name(inArrayType);
    }
    else
    {
        kernel_name += "_op";
        kernel_name += rtc_array_type_name(inArrayType);
        kernel_name += rtc_array_type_name(outArrayType);
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
        if(largeTwdBatchIsTransformCount)
            kernel_name += "_batchcount";
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

    // callback kernels need to disable buffer load/store
    if(enable_callbacks || dir2regMode == DirectRegType::FORCE_OFF_OR_NOT_SUPPORT)
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;

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
                         bool                          largeTwdBatchIsTransformCount,
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

    std::vector<unsigned int> all_factors;

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

        // get all factors by concat two vectors
        all_factors = kernel.kernel0.factors;
        all_factors.insert(
            all_factors.end(), kernel.kernel1.factors.begin(), kernel.kernel1.factors.end());
    }
    else
    {
        std::unique_ptr<StockhamKernel> kernel;
        if(scheme == CS_KERNEL_STOCKHAM)
            kernel = std::make_unique<StockhamKernelRR>(specs);
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            kernel = std::make_unique<StockhamKernelCC>(specs, largeTwdBatchIsTransformCount);
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

        // get factors vector
        all_factors = kernel->factors;
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
    std::string src;
    src += common_h;
    src += memory_gfx_h;
    src += callback_h;
    src += butterfly_constant_h;

    // only SBCCs need this
    if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        src += large_twiddles_h;
    // append the neccessary functions only
    append_radix_h(src, all_factors);
    // SBCCs don't need this
    if(scheme != CS_KERNEL_STOCKHAM_BLOCK_CC)
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
    src += rtc_precision_type_decl(precision);
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

    src += rtc_const_cbtype_decl(enable_callbacks);

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

    // callback kernels need to disable buffer load/store
    if(enable_callbacks || dir2regMode == DirectRegType::FORCE_OFF_OR_NOT_SUPPORT)
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;

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
    return src;
}
