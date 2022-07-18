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

#ifndef RTC_STOCKHAM
#define RTC_STOCKHAM

#include <vector>

#include "../device/generator/stockham_gen.h"
#include "../device/kernels/common.h"
#include "compute_scheme.h"
#include "rocfft.h"
#include "rtc.h"

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
                                     SBRC_TRANSPOSE_TYPE     transpose_type,
                                     bool                    enable_callbacks,
                                     bool                    enable_scaling);

// generate source for RTC stockham kernel
std::string stockham_rtc(const StockhamGeneratorSpecs& specs,
                         const StockhamGeneratorSpecs& specs2d,
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
                         SBRC_TRANSPOSE_TYPE           transpose_type,
                         bool                          enable_callbacks,
                         bool                          enable_scaling);

struct RTCKernelStockham : public RTCKernel
{
    RTCKernelStockham(const std::string& kernel_name, const std::vector<char>& code)
        : RTCKernel(kernel_name, code)
    {
    }

    static RTCKernel::RTCGenerator generate_from_node(const TreeNode&    node,
                                                      const std::string& gpu_arch,
                                                      bool               enable_callbacks);

    virtual RTCKernelArgs get_launch_args(DeviceCallIn& data) override;
};

#endif
