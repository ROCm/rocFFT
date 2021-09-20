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

#ifndef ROCFFT_RTC_H
#define ROCFFT_RTC_H

#include "rocfft.h"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <memory>
#include <string>
#include <vector>

struct DeviceCallIn;
class TreeNode;

struct RTCKernel
{
    // try to compile kernel for node, and attach compiled kernel to
    // node if successful.  returns nullptr if there is no matching
    // supported scheme + problem size.  throws runtime_error on
    // error.
    static std::unique_ptr<RTCKernel>
        runtime_compile(TreeNode& node, const char* gpu_arch, bool enable_callbacks = false);

    ~RTCKernel()
    {
        kernel = nullptr;
        (void)hipModuleUnload(module);
        module = nullptr;
    }

    // disallow copies, since we expect this to be managed by smart ptr
    RTCKernel(const RTCKernel&) = delete;
    RTCKernel(RTCKernel&&)      = delete;
    void operator=(const RTCKernel&) = delete;

    void launch(DeviceCallIn& data);

    // close kernel cache explicitly.  cache will be reopened
    // on-demand.
    static void close_cache();

private:
    // private ctor, use "runtime_compile" to build kernel for a node
    RTCKernel(const std::string& kernel_name, const std::vector<char>& code);
    hipModule_t   module = nullptr;
    hipFunction_t kernel = nullptr;

    // compile source to a code object
    static std::vector<char> compile(const std::string& kernel_src);
};

#endif
