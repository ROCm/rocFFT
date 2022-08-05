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

#ifndef ROCFFT_RTC_H
#define ROCFFT_RTC_H

#include "rocfft.h"
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct DeviceCallIn;
class TreeNode;

// Helper class that handles alignment of kernel arguments
class RTCKernelArgs
{
public:
    RTCKernelArgs() = default;
    void append_ptr(void* ptr)
    {
        append(&ptr, sizeof(void*));
    }
    void append_size_t(size_t s)
    {
        append(&s, sizeof(size_t));
    }
    void append_unsigned_int(unsigned int i)
    {
        append(&i, sizeof(unsigned int));
    }
    void append_double(double d)
    {
        append(&d, sizeof(double));
    }
    void append_float(float f)
    {
        append(&f, sizeof(float));
    }

    size_t size_bytes() const
    {
        return buf.size();
    }
    void* data()
    {
        return buf.data();
    }

private:
    void append(void* src, size_t nbytes)
    {
        // values need to be aligned to their width (i.e. 8-byte values
        // need 8-byte alignment, 4-byte needs 4-byte alignment)
        size_t oldsize = buf.size();
        size_t padding = oldsize % nbytes ? nbytes - (oldsize % nbytes) : 0;
        buf.resize(oldsize + padding + nbytes);
        std::copy_n(static_cast<const char*>(src), nbytes, buf.begin() + oldsize + padding);
    }
    std::vector<char> buf;
};

// Base class for a runtime compiled kernel.  Subclassed for
// different kernel types that each have their own details about how
// to be launched.
struct RTCKernel
{
    // try to compile kernel for node, and attach compiled kernel to
    // node if successful.  returns nullptr if there is no matching
    // supported scheme + problem size.  throws runtime_error on
    // error.
    static std::shared_future<std::unique_ptr<RTCKernel>> runtime_compile(
        const TreeNode& node, const std::string& gpu_arch, bool enable_callbacks = false);

    virtual ~RTCKernel()
    {
        kernel = nullptr;
        (void)hipModuleUnload(module);
        module = nullptr;
    }

    // disallow copies, since we expect this to be managed by smart ptr
    RTCKernel(const RTCKernel&) = delete;
    RTCKernel(RTCKernel&&)      = delete;
    void operator=(const RTCKernel&) = delete;

    // normal launch from within rocFFT execution plan
    void launch(DeviceCallIn& data);
    // direct launch with kernel args
    void launch(RTCKernelArgs& kargs,
                dim3           gridDim,
                dim3           blockDim,
                unsigned int   lds_bytes,
                hipStream_t    stream = nullptr);

    // Subclasses implement this - each kernel type has different
    // parameters
    virtual RTCKernelArgs get_launch_args(DeviceCallIn& data) = 0;

    // runtime_compile dispatches to subclasses, those subclasses
    // return callables to do the code generation, so that the
    // compilation and caching can live in the cached_compile method.
    //
    // function to generate the name of a kernel, given no arguments
    using kernel_name_gen_t = std::function<std::string()>;
    // functor to generate the source code of a kernel, given the
    // kernel name.  but remember the source in case we're asked to
    // generate it again.
    struct kernel_src_gen_t
    {
        using generator_func = std::function<std::string(const std::string&)>;

        // allow default ctor + assign, plus ctor + assign from function
        kernel_src_gen_t() = default;
        kernel_src_gen_t(generator_func f)
            : f(f)
        {
        }
        kernel_src_gen_t& operator=(const kernel_src_gen_t&) = default;
        kernel_src_gen_t& operator                           =(generator_func f)
        {
            this->f = f;
            kernel_src.clear();
            return *this;
        }
        // forward a call to the function, if we don't already have the source
        std::string operator()(const std::string& kernel_name)
        {
            if(kernel_src.empty())
                kernel_src = f(kernel_name);
            return kernel_src;
        }
        operator bool() const
        {
            return static_cast<bool>(f);
        }

    private:
        generator_func f;
        std::string    kernel_src;
    };
    // function to construct the correct RTCKernel object, given a kernel name and its compiled code
    using rtckernel_construct_t
        = std::function<std::unique_ptr<RTCKernel>(const std::string&, const std::vector<char>&)>;

    // Get compiled code object for a kernel.  Checks the cache to
    // see if the kernel has already been compiled and returns the
    // cached kernel if present.
    //
    // Otherwise, calls "generate_src" to generate the source, compiles
    // the source, and updates the cache before returning the compiled
    // kernel.  Tries in-process compile
    // first and falls back to subprocess if necessary.
    static std::vector<char> cached_compile(const std::string& kernel_name,
                                            const std::string& gpu_arch,
                                            kernel_src_gen_t   generate_src);

    // compile source to a code object, in the current process.
    static std::vector<char> compile_inprocess(const std::string& kernel_src,
                                               const std::string& gpu_arch);

protected:
    // protected ctor, use "runtime_compile" to build kernel for a node
    RTCKernel(const std::string& kernel_name, const std::vector<char>& code);

    struct RTCGenerator
    {
        kernel_name_gen_t     generate_name;
        kernel_src_gen_t      generate_src;
        rtckernel_construct_t construct_rtckernel;
        bool                  valid() const
        {
            return generate_name && generate_src && construct_rtckernel;
        }
    };

    hipModule_t   module = nullptr;
    hipFunction_t kernel = nullptr;

private:
    // Lock for in-process compilation - due to limits in ROCclr, we
    // can do at most one compilation in a process before we have to
    // delegate to a subprocess.  But we should at least do one
    // in-process instead of making everything go to subprocess.
    static std::mutex        compile_lock;
    static std::vector<char> compile_subprocess(const std::string& kernel_src,
                                                const std::string& gpu_arch);
};

#endif
