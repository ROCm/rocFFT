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

#include "../../shared/arithmetic.h"
#include "../../shared/gpubuf.h"
#include "device/generator/stockham_gen.h"
#include "rtc_compile.h"
#include "rtc_stockham_gen.h"
#include "rtc_stockham_kernel.h"

#include <iterator>
#include <random>
#include <set>

static const std::vector<unsigned int> supported_factors
    = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};
static const std::vector<unsigned int> supported_wgs{64, 128, 256};

// recursively find all unique factorizations of given length.  each
// factorization is a vector of ints, sorted so they're uniquified in
// a set.
std::set<std::vector<unsigned int>> factorize(unsigned int length)
{
    std::set<std::vector<unsigned int>> ret;
    for(auto factor : supported_factors)
    {
        if(length % factor == 0)
        {
            unsigned int remain = length / factor;
            if(remain == 1)
                ret.insert({factor});
            else
            {
                // recurse into remainder
                auto remain_factorization = factorize(remain);
                for(auto& remain_factors : remain_factorization)
                {
                    std::vector<unsigned int> factors{factor};
                    std::copy(
                        remain_factors.begin(), remain_factors.end(), std::back_inserter(factors));
                    std::sort(factors.begin(), factors.end());
                    ret.insert(factors);
                }
            }
        }
    }
    return ret;
}

// recursively return power set of a range of ints
std::set<std::vector<unsigned int>> power_set(std::vector<unsigned int>::const_iterator begin,
                                              std::vector<unsigned int>::const_iterator end)
{
    std::set<std::vector<unsigned int>> ret;
    // either include the front element in the output, or don't
    if(std::distance(begin, end) == 1)
    {
        ret.insert({*begin});
        ret.insert({});
    }
    else
    {
        // recurse into the remainder
        auto remain = power_set(begin + 1, end);
        for(auto r : remain)
        {
            ret.insert(r);
            r.push_back(*begin);
            ret.insert(r);
        }
    }
    return ret;
}

std::set<unsigned int>
    supported_threads_per_transform(const std::vector<unsigned int>& factorization)
{
    std::set<unsigned int> tpts;
    auto                   tpt_candidates = power_set(factorization.begin(), factorization.end());
    for(auto tpt : tpt_candidates)
    {
        if(tpt.empty())
            continue;
        auto product = std::accumulate(tpt.begin(), tpt.end(), 1U, std::multiplies<unsigned int>());
        tpts.insert(product);
    }
    return tpts;
}

std::string test_kernel_name(unsigned int                     length,
                             const std::vector<unsigned int>& factorization,
                             unsigned int                     wgs,
                             unsigned int                     tpt,
                             bool                             half_lds,
                             bool                             direct_to_from_reg)
{
    std::string ret = "fft_test_len_";
    ret += std::to_string(length);
    ret += "_factors";
    for(auto f : factorization)
    {
        ret += "_";
        ret += std::to_string(f);
    }
    ret += "_wgs_";
    ret += std::to_string(wgs);
    ret += "_tpt_";
    ret += std::to_string(tpt);
    if(half_lds)
        ret += "_halfLds";
    if(direct_to_from_reg)
        ret += "_dirReg";

    return ret;
}

std::string test_kernel_src(const std::string&               kernel_name,
                            unsigned int&                    transforms_per_block,
                            unsigned int                     length,
                            const std::vector<unsigned int>& factorization,
                            unsigned int                     wgs,
                            unsigned int                     tpt,
                            bool                             half_lds,
                            bool                             direct_to_from_reg)
{
    StockhamGeneratorSpecs specs{factorization,
                                 {},
                                 {static_cast<unsigned int>(rocfft_precision_single)},
                                 wgs,
                                 "CS_KERNEL_STOCKHAM"};
    specs.threads_per_transform = tpt;
    specs.half_lds              = half_lds;
    specs.direct_to_from_reg    = direct_to_from_reg;

    return stockham_rtc(specs,
                        specs,
                        &transforms_per_block,
                        kernel_name,
                        CS_KERNEL_STOCKHAM,
                        -1,
                        rocfft_precision_single,
                        rocfft_placement_notinplace,
                        rocfft_array_type_complex_interleaved,
                        rocfft_array_type_complex_interleaved,
                        true,
                        0,
                        0,
                        false,
                        EmbeddedType::NONE,
                        direct_to_from_reg ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                                           : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,
                        IntrinsicAccessType::DISABLE_BOTH,
                        SBRC_TRANSPOSE_TYPE::NONE,
                        CallbackType::NONE,
                        BluesteinFuseType::BFT_NONE,
                        {},
                        {});
}

// things that we need to remember between kernel launches
struct device_data_t
{
    std::vector<rocfft_complex<float>> host_input_buf;
    gpubuf_t<rocfft_complex<float>>    fake_twiddles;
    gpubuf_t<rocfft_complex<float>>    input_buf;
    gpubuf_t<rocfft_complex<float>>    output_buf;
    gpubuf_t<size_t>                   lengths;
    gpubuf_t<size_t>                   stride_in;
    gpubuf_t<size_t>                   stride_out;
    size_t                             batch;
    hipEvent_t                         start;
    hipEvent_t                         stop;

    device_data_t()
    {
        if(hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
            throw std::runtime_error("hipEventCreate failed");
    }
    ~device_data_t()
    {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
};

// run the kernel, returning the median execution time
float launch_kernel(RTCKernel&     kernel,
                    unsigned int   blocks,
                    unsigned int   wgs,
                    unsigned int   lds_bytes,
                    device_data_t& data)
{
    RTCKernelArgs kargs;
    kargs.append_ptr(data.fake_twiddles.data());
    kargs.append_size_t(1);
    kargs.append_ptr(data.lengths.data());
    kargs.append_ptr(data.stride_in.data());
    kargs.append_ptr(data.stride_out.data());
    kargs.append_size_t(data.batch);
    kargs.append_size_t(0);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_unsigned_int(0);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(data.input_buf.data());
    kargs.append_ptr(data.output_buf.data());
    std::vector<float> times;
    for(unsigned int i = 0; i < 11; ++i)
    {
        // simulate rocfft-bench behaviour - memcpy input to device
        // before each execution
        if(hipMemcpy(data.input_buf.data(),
                     data.host_input_buf.data(),
                     data.host_input_buf.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");

        if(hipEventRecord(data.start) != hipSuccess)
            throw std::runtime_error("hipEventRecord start failed");
        kernel.launch(kargs, {blocks}, {wgs}, lds_bytes);
        if(hipEventRecord(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord stop failed");
        if(hipEventSynchronize(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
        float time;
        if(hipEventElapsedTime(&time, data.start, data.stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");
        times.push_back(time);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

unsigned int get_lds_bytes(unsigned int length, unsigned int transforms_per_block, bool half_lds)
{
    // assume single precision complex
    return length * transforms_per_block * sizeof(rocfft_complex<float>) / (half_lds ? 2 : 1);
}

size_t batch_size(unsigned int length)
{
    // target 2 GiB memory usage (2^31), assume single precision so
    // each element is 2^3 bytes
    size_t target_elems = 1U << 28;
    return target_elems / length;
}

std::vector<rocfft_complex<float>> create_input_buf(unsigned int length, size_t batch)
{
    auto                               elems = length * batch;
    std::vector<rocfft_complex<float>> buf;
    buf.reserve(elems);
    std::mt19937 gen;
    for(unsigned int i = 0; i < elems; ++i)
    {
        float x = static_cast<float>(gen()) / static_cast<float>(gen.max());
        float y = static_cast<float>(gen()) / static_cast<float>(gen.max());
        buf.push_back({x, y});
    }
    return buf;
}

gpubuf_t<rocfft_complex<float>> create_device_buf(unsigned int length, size_t batch)
{
    auto                            elems = length * batch;
    gpubuf_t<rocfft_complex<float>> device_buf;
    if(device_buf.alloc(elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemset(device_buf.data(), 0, elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMemset");

    return device_buf;
}

gpubuf_t<size_t> create_lengths(unsigned int length)
{
    gpubuf_t<size_t> device_buf;
    if(device_buf.alloc(sizeof(size_t)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");

    if(hipMemcpy(device_buf.data(), &length, sizeof(size_t), hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

gpubuf_t<size_t> create_strides(unsigned int length)
{
    std::array<size_t, 2> strides{1, length};
    gpubuf_t<size_t>      device_buf;
    if(device_buf.alloc(sizeof(size_t) * 2) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(device_buf.data(), strides.data(), 2 * sizeof(size_t), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        puts("usage: brute_force_1d_opt length");
        return 1;
    }

    unsigned int length = std::stoul(argv[1]);

    if(hipInit(0) != hipSuccess)
        throw std::runtime_error("hipInit failure");

    hipDeviceProp_t device_prop;
    if(hipGetDeviceProperties(&device_prop, 0) != hipSuccess)
        throw std::runtime_error("hipGetDeviceProperties failure");

    // init device data
    device_data_t data;
    data.batch = batch_size(length);
    // construct random input on host side, allocate input/output
    // buffers on GPU.  input will be copied to GPU at launch time
    data.host_input_buf = create_input_buf(length, data.batch);
    data.input_buf      = create_device_buf(length, data.batch);
    data.output_buf     = create_device_buf(length, data.batch);
    // create twiddles table same length as FFT.  this isn't exactly
    // what rocFFT would do but is close enough.
    auto host_twiddles = create_input_buf(length, 1);
    data.fake_twiddles = create_device_buf(length, 1);
    if(hipMemcpy(data.fake_twiddles.data(),
                 host_twiddles.data(),
                 host_twiddles.size() * sizeof(rocfft_complex<float>),
                 hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    data.lengths    = create_lengths(length);
    data.stride_in  = create_strides(length);
    data.stride_out = create_strides(length);
    printf("length %u, batch %zu\n", length, data.batch);
    fflush(stdout);

    const auto factorizations = factorize(length);

    // remember the best configuration observed so far
    float                     best_time               = std::numeric_limits<float>::max();
    unsigned int              best_wgs                = 0;
    unsigned int              best_tpt                = 0;
    bool                      best_half_lds           = true;
    bool                      best_direct_to_from_reg = true;
    std::vector<unsigned int> best_factorization;
    std::string               best_kernel_src;

    for(auto factorization : factorizations)
    {
        auto tpts = supported_threads_per_transform(factorization);

        // go through all permutations of the factors
        do
        {
            for(auto wgs : supported_wgs)
            {
                for(auto tpt : tpts)
                {
                    if(tpt < wgs)
                    {
                        for(bool half_lds : {true, false})
                        {
                            for(bool direct_to_from_reg : {true, false})
                            {
                                // half lds currently requires direct to/from reg
                                if(half_lds && !direct_to_from_reg)
                                    continue;
                                auto kernel_name = test_kernel_name(
                                    length, factorization, wgs, tpt, half_lds, direct_to_from_reg);
                                unsigned int transforms_per_block = 0;
                                auto         kernel_src           = test_kernel_src(kernel_name,
                                                                  transforms_per_block,
                                                                  length,
                                                                  factorization,
                                                                  wgs,
                                                                  tpt,
                                                                  half_lds,
                                                                  direct_to_from_reg);

                                auto code = compile_inprocess(kernel_src, device_prop.gcnArchName);
                                RTCKernelStockham kernel(kernel_name, code);

                                float time = launch_kernel(
                                    kernel,
                                    DivRoundingUp<unsigned int>(data.batch, transforms_per_block),
                                    tpt * transforms_per_block,
                                    get_lds_bytes(length, transforms_per_block, half_lds),
                                    data);

                                // print median time for this length
                                // in a format that can be easily
                                // grepped for and shoved into a
                                // database if desired
                                printf("%u,%s,%.5f\n",
                                       length,
                                       kernel_name.c_str(),
                                       static_cast<double>(time));
                                fflush(stdout);

                                if(time < best_time)
                                {
                                    best_time               = time;
                                    best_wgs                = wgs;
                                    best_tpt                = tpt;
                                    best_half_lds           = half_lds;
                                    best_direct_to_from_reg = direct_to_from_reg;
                                    best_factorization      = factorization;
                                    best_kernel_src         = std::move(kernel_src);
                                }
                            }
                        }
                    }
                }
            }
        } while(std::next_permutation(factorization.begin(), factorization.end()));
    }

    // print a line with the best config, to go into kernel-generator.py
    printf("  NS(length=% 4d, workgroup_size=% 3d, threads_per_transform=% 3d, factors=(",
           length,
           best_wgs,
           best_tpt);
    bool first_factor = true;
    for(auto f : best_factorization)
    {
        if(!first_factor)
            printf(", ");
        first_factor = false;
        printf("%u", f);
    }
    printf(")");
    if(!best_half_lds)
        printf(", half_lds=False");
    if(!best_direct_to_from_reg)
        printf(", direct_to_from_reg=False");
    printf("),\n");
    fflush(stdout);

    return 0;
}
