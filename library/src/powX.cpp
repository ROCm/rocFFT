// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "rocfft.h"

#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "transform.h"

#include "radix_table.h"

#include "kernel_launch.h"

#include "function_pool.h"
#include "ref_cpu.h"

#include "real2complex.h"

#include "../../shared/printbuffer.h"
#include "rocfft_hip.h"

std::atomic<bool> fn_checked(false);

SBRC_TRANSPOSE_TYPE sbrc_3D_transpose_type(unsigned int               blockWidth,
                                           size_t                     alignment_dimension,
                                           const std::vector<size_t>& length)
{
    if(is_diagonal_sbrc_3D_length(length.front()) && is_cube_size(length))
        return DIAGONAL;
    if(alignment_dimension % blockWidth == 0)
        return TILE_ALIGNED;
    return TILE_UNALIGNED;
}

// This function is called during creation of plan: enqueue the HIP kernels by function
// pointers. Return true if everything goes well. Any internal device memory allocation
// failure returns false right away.
bool PlanPowX(ExecPlan& execPlan)
{
    for(const auto& node : execPlan.execSeq)
    {
        if(node->scheme == CS_KERNEL_STOCKHAM)
        {
            auto kernel    = function_pool::get_kernel(fpkey(node->length[0], node->precision));
            node->twiddles = twiddles_create(node->length[0],
                                             node->precision,
                                             false,
                                             LTWD_BASE_DEFAULT,
                                             false,
                                             false,
                                             kernel.factors);
            if(node->twiddles == nullptr)
                return false;
        }
        else if(node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            auto kernel = function_pool::get_kernel(
                fpkey(node->length[0], node->precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
            node->twiddles = twiddles_create(node->length[0],
                                             node->precision,
                                             false,
                                             LTWD_BASE_DEFAULT,
                                             false,
                                             false,
                                             kernel.factors);
            if(node->twiddles == nullptr)
                return false;
        }
        else if((node->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                || (node->scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)
                || (node->scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY))
        {
            node->twiddles = twiddles_create(
                node->length[0], node->precision, false, LTWD_BASE_DEFAULT, false, false);
            if(node->twiddles == nullptr)
                return false;
        }
        else if((node->scheme == CS_KERNEL_R_TO_CMPLX)
                || (node->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE)
                || (node->scheme == CS_KERNEL_CMPLX_TO_R))
        {
            node->twiddles = twiddles_create(
                2 * node->length[0], node->precision, false, LTWD_BASE_DEFAULT, true, false);
            if(node->twiddles == nullptr)
                return false;
        }
        // need twiddles of the lowest dimension after the transpose is done
        else if(node->scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        {
            // C2R transform ends up getting shorter by 1 along that dimension also
            node->twiddles = twiddles_create(2 * (node->length.back() - 1),
                                             node->precision,
                                             false,
                                             LTWD_BASE_DEFAULT,
                                             true,
                                             false);
            if(node->twiddles == nullptr)
                return false;
        }
        else if(node->scheme == CS_KERNEL_2D_SINGLE)
        {
            // create one set of twiddles for each dimension
            node->twiddles = twiddles_create_2D(node->length[0], node->length[1], node->precision);
        }
        else if(node->scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
        {
            node->twiddles = twiddles_create(
                node->length[0], node->precision, false, LTWD_BASE_DEFAULT, false, true);
            if(node->twiddles == nullptr)
                return false;
        }

        if(node->large1D != 0)
        {
            node->twiddles_large = twiddles_create(
                node->large1D, node->precision, true, node->largeTwdBase, false, false);
            if(node->twiddles_large == nullptr)
                return false;
        }
    }
    // copy host buffer to device buffer
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        execPlan.execSeq[i]->devKernArg = kargs_create(execPlan.execSeq[i]->length,
                                                       execPlan.execSeq[i]->inStride,
                                                       execPlan.execSeq[i]->outStride,
                                                       execPlan.execSeq[i]->iDist,
                                                       execPlan.execSeq[i]->oDist);
        if(execPlan.execSeq[i]->devKernArg == nullptr)
            return false;
    }

    if(!fn_checked)
    {
        fn_checked = true;
        function_pool::verify_no_null_functions();
    }

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DevFnCall ptr = nullptr;
        GridParam gp;
        size_t    bwd = 1;
        size_t    wgs, lds, lds_padding;
        wgs = lds = lds_padding = 0;

        switch(execPlan.execSeq[i]->scheme)
        {
        case CS_KERNEL_STOCKHAM:
        {
            // get working group size and number of transforms
            size_t batch = execPlan.execSeq[i]->batch;
            for(size_t j = 1; j < execPlan.execSeq[i]->length.size(); j++)
                batch *= execPlan.execSeq[i]->length[j];

            auto kernel = function_pool::get_kernel(
                fpkey(execPlan.execSeq[i]->length[0], execPlan.execSeq[0]->precision));

            ptr = kernel.device_function;
            if(kernel.threads_per_block > 0)
            {
                gp.b_x   = (batch + kernel.batches_per_block - 1) / kernel.batches_per_block;
                gp.tpb_x = kernel.threads_per_block;

                lds = execPlan.execSeq[i]->length[0] * kernel.batches_per_block;
            }
            else
            {
                size_t workGroupSize;
                size_t numTransforms;
                GetWGSAndNT(execPlan.execSeq[i]->length[0], workGroupSize, numTransforms);
                gp.b_x   = (batch % numTransforms) ? 1 + (batch / numTransforms)
                                                   : (batch / numTransforms);
                gp.tpb_x = workGroupSize;
            }
        }
        break;
        case CS_KERNEL_STOCKHAM_BLOCK_CC:
        {
            auto kernel = function_pool::get_kernel(fpkey(execPlan.execSeq[i]->length[0],
                                                          execPlan.execSeq[0]->precision,
                                                          CS_KERNEL_STOCKHAM_BLOCK_CC));
            ptr         = kernel.device_function;
            gp.b_x      = ((execPlan.execSeq[i]->length[1]) - 1) / kernel.batches_per_block + 1;
            // repeat for higher dimensions + batch
            gp.b_x *= std::accumulate(execPlan.execSeq[i]->length.begin() + 2,
                                      execPlan.execSeq[i]->length.end(),
                                      execPlan.execSeq[i]->batch,
                                      std::multiplies<size_t>());
            gp.tpb_x = kernel.threads_per_block;

            lds = execPlan.execSeq[i]->length[0] * kernel.batches_per_block;
        }
        break;
        case CS_KERNEL_STOCKHAM_BLOCK_RC:
            ptr = function_pool::get_function(fpkey(execPlan.execSeq[i]->length[0],
                                                    execPlan.execSeq[0]->precision,
                                                    CS_KERNEL_STOCKHAM_BLOCK_RC));
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = (execPlan.execSeq[i]->length[1]) / bwd;
            // repeat for higher dimensions + batch
            gp.b_x *= std::accumulate(execPlan.execSeq[i]->length.begin() + 2,
                                      execPlan.execSeq[i]->length.end(),
                                      execPlan.execSeq[i]->batch,
                                      std::multiplies<size_t>());
            gp.tpb_x = wgs;
            break;
        case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        {
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            // each block handles 'bwd' rows
            gp.b_x = DivRoundingUp(execPlan.execSeq[i]->length[2], bwd)
                     * execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;
            gp.tpb_x = wgs;

            auto transposeType = sbrc_3D_transpose_type(
                bwd, execPlan.execSeq[i]->length[2], execPlan.execSeq[i]->length);

            ptr = function_pool::get_function(fpkey(execPlan.execSeq[i]->length[0],
                                                    execPlan.execSeq[0]->precision,
                                                    CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
                                                    transposeType));
            break;
        }
        case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        {
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            auto transposeType = sbrc_3D_transpose_type(bwd,
                                                        execPlan.execSeq[i]->length[1]
                                                            * execPlan.execSeq[i]->length[2],
                                                        execPlan.execSeq[i]->length);

            ptr = function_pool::get_function(fpkey(execPlan.execSeq[i]->length[0],
                                                    execPlan.execSeq[0]->precision,
                                                    execPlan.execSeq[i]->scheme,
                                                    transposeType));

            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = std::accumulate(execPlan.execSeq[i]->length.begin() + 1,
                                     execPlan.execSeq[i]->length.end(),
                                     execPlan.execSeq[i]->batch,
                                     std::multiplies<size_t>());
            // do 'bwd' rows per block
            gp.b_x /= bwd;
            gp.tpb_x = wgs;
            break;
        }
        case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
        {
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            auto transposeType = sbrc_3D_transpose_type(bwd,
                                                        execPlan.execSeq[i]->length[1]
                                                            * execPlan.execSeq[i]->length[2],
                                                        execPlan.execSeq[i]->length);

            ptr = function_pool::get_function(fpkey(execPlan.execSeq[i]->length[0],
                                                    execPlan.execSeq[0]->precision,
                                                    execPlan.execSeq[i]->scheme,
                                                    transposeType));

            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = std::accumulate(execPlan.execSeq[i]->length.begin() + 1,
                                     execPlan.execSeq[i]->length.end(),
                                     execPlan.execSeq[i]->batch,
                                     std::multiplies<size_t>());
            // do 'bwd' rows per block
            gp.b_x /= bwd;
            gp.tpb_x    = wgs;
            lds_padding = 1; // 1 element padding per row for even-length real2complx usage
            break;
        }
        case CS_KERNEL_TRANSPOSE:
        case CS_KERNEL_TRANSPOSE_XY_Z:
        case CS_KERNEL_TRANSPOSE_Z_XY:
            ptr      = &FN_PRFX(transpose_var2);
            gp.tpb_x = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 64;
            gp.tpb_y = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 16;
            break;
        case CS_KERNEL_COPY_R_TO_CMPLX:
            ptr      = &real2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_R:
            ptr      = &complex2real;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_HERM_TO_CMPLX:
            ptr      = &hermitian2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
            ptr      = &complex2hermitian;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_R_TO_CMPLX:
            ptr = &r2c_1d_post;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_R_TO_CMPLX_TRANSPOSE:
            ptr = &r2c_1d_post_transpose;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CMPLX_TO_R:
            ptr = &c2r_1d_pre;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_TRANSPOSE_CMPLX_TO_R:
            ptr = &transpose_c2r_1d_pre;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CHIRP:
            ptr      = &FN_PRFX(chirp);
            gp.tpb_x = 64;
            break;
        case CS_KERNEL_PAD_MUL:
        case CS_KERNEL_FFT_MUL:
        case CS_KERNEL_RES_MUL:
            ptr      = &FN_PRFX(mul);
            gp.tpb_x = 64;
            break;
        case CS_KERNEL_2D_SINGLE:
        {
            auto kernel = function_pool::get_kernel(fpkey(execPlan.execSeq[i]->length[0],
                                                          execPlan.execSeq[i]->length[1],
                                                          execPlan.execSeq[0]->precision));
            ptr         = kernel.device_function;
            if(!kernel.factors.empty())
            {
                // when old generator goes away, we will always have factors
                gp.b_x = (execPlan.execSeq[i]->batch + kernel.batches_per_block - 1)
                         / kernel.batches_per_block;
                gp.tpb_x = kernel.threads_per_block;
                lds      = execPlan.execSeq[i]->length[0] * execPlan.execSeq[i]->length[1]
                      * kernel.batches_per_block;
            }
            else
            {
                // Run one threadblock per transform, since we're
                // combining a row transform and a column transform in
                // one kernel.  The transform must not cross threadblock
                // boundaries, or else we are unable to make the row
                // transform finish completely before starting the column
                // transform.
                gp.b_x   = execPlan.execSeq[i]->batch;
                gp.tpb_x = Get2DSingleThreadCount(
                    execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->length[1], GetWGSAndNT);
            }
            // if we're doing 3D transform, we need to repeat the 2D
            // transform in the 3rd dimension
            if(execPlan.execSeq[i]->length.size() > 2)
                gp.b_x *= execPlan.execSeq[i]->length[2];
            break;
        }
        case CS_KERNEL_APPLY_CALLBACK:
            ptr      = apply_real_callback;
            gp.tpb_x = 64;
            break;
        default:
            rocfft_cout << "should not be in this case" << std::endl;
            rocfft_cout << "scheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
            assert(false);
        }

        gp.lds_bytes = (lds + lds_padding * bwd) * PrecisionWidth(execPlan.execSeq[0]->precision)
                       * sizeof(float2);
        execPlan.devFnCall.push_back(ptr);
        execPlan.gridParam.push_back(gp);
    }

    return true;
}

static size_t data_size_bytes(const std::vector<size_t>& lengths,
                              rocfft_precision           precision,
                              rocfft_array_type          type)
{
    // first compute the raw number of elements
    size_t elems = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>());
    // size of each element
    size_t elemsize = (precision == rocfft_precision_single ? sizeof(float) : sizeof(double));
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
        // complex needs two numbers per element
        return 2 * elems * elemsize;
    case rocfft_array_type_real:
        // real needs one number per element
        return elems * elemsize;
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    {
        // hermitian requires 2 numbers per element, but innermost
        // dimension is cut down to roughly half
        size_t non_innermost = elems / lengths[0];
        return 2 * non_innermost * elemsize * ((lengths[0] / 2) + 1);
    }
    case rocfft_array_type_unset:
        // we should really have an array type at this point
        assert(false);
        return 0;
    }
}

static float execution_bandwidth_GB_per_s(size_t data_size_bytes, float duration_ms)
{
    // divide bytes by (1000000 * milliseconds) to get GB/s
    return static_cast<float>(data_size_bytes) / (1000000.0 * duration_ms);
}

// NOTE: HIP returns the maximum global frequency in kHz, which might
// not be the actual frequency when the transform ran.  This function
// might also return 0.0 if the bandwidth can't be queried.
static float max_memory_bandwidth_GB_per_s()
{
    // Try to get the device bandwidth from an environment variable:
    char* pdevbw = NULL;
    pdevbw       = getenv("ROCFFT_DEVICE_BW");
    if(pdevbw != NULL)
    {
        return atof(pdevbw);
    }

    // Try to get the device bandwidth from hip calls:
    int deviceid = 0;
    hipGetDevice(&deviceid);
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;
    hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, deviceid);
    hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, deviceid);
    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1000.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
}

// work out the buffer size, based on the largest size implied by
// dist or strides
//
// FIXME: share this with client_utils somehow
size_t compute_size(const std::vector<size_t>& length,
                    const std::vector<size_t>& stride,
                    size_t                     nbatch,
                    size_t                     dist)
{
    size_t val = nbatch * dist;
    for(int i = 0; i < length.size(); ++i)
    {
        val = std::max(val, length[i] * stride[i]);
    }
    return val;
}

// Print either an input or output buffer, given column-major dimensions
void DebugPrintBuffer(rocfft_ostream&            stream,
                      rocfft_array_type          type,
                      rocfft_precision           precision,
                      void*                      buffer[],
                      const std::vector<size_t>& length_cm,
                      const std::vector<size_t>& stride_cm,
                      size_t                     dist,
                      size_t                     offset,
                      size_t                     batch)
{
    const size_t size_elems = compute_size(length_cm, stride_cm, batch, dist);

    size_t base_type_size = (precision == rocfft_precision_double) ? sizeof(double) : sizeof(float);
    if(type != rocfft_array_type_real)
    {
        // complex elements
        base_type_size *= 2;
    }

    size_t size_bytes = size_elems * base_type_size;
    // convert length, stride to row-major for use with printbuffer
    auto length_rm = length_cm;
    auto stride_rm = stride_cm;
    std::reverse(length_rm.begin(), length_rm.end());
    std::reverse(stride_rm.begin(), stride_rm.end());
    std::vector<std::vector<char>> bufvec;
    std::vector<size_t>            print_offset(2, offset);
    if(array_type_is_planar(type))
    {
        // separate the real/imag data, so printbuffer will print them separately
        bufvec.resize(2);
        bufvec.front().resize(size_bytes / 2);
        bufvec.back().resize(size_bytes / 2);
        hipMemcpy(bufvec.front().data(), buffer[0], size_bytes / 2, hipMemcpyDeviceToHost);
        hipMemcpy(bufvec.back().data(), buffer[1], size_bytes / 2, hipMemcpyDeviceToHost);

        printbuffer(
            precision, type, bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
    }
    else
    {
        bufvec.resize(1);
        bufvec.front().resize(size_bytes);
        hipMemcpy(bufvec.front().data(), buffer[0], size_bytes, hipMemcpyDeviceToHost);
        printbuffer(
            precision, type, bufvec, length_rm, stride_rm, batch, dist, print_offset, stream);
    }
}

// for callbacks, work out which nodes of the plan are loading data
// from global memory, and storing data to global memory
static std::pair<TreeNode*, TreeNode*> get_load_store_nodes(const ExecPlan& execPlan)
{
    const auto& seq = execPlan.execSeq;

    // look forward for the first node that reads from input
    auto      load_it = std::find_if(seq.begin(), seq.end(), [&](const TreeNode* n) {
        return n->obIn == execPlan.rootPlan->obIn;
    });
    TreeNode* load    = load_it == seq.end() ? nullptr : *load_it;

    // look backward for the last node that writes to output
    auto      store_it = std::find_if(seq.rbegin(), seq.rend(), [&](const TreeNode* n) {
        return n->obOut == execPlan.rootPlan->obOut;
    });
    TreeNode* store    = store_it == seq.rend() ? nullptr : *store_it;

    assert(load && store);
    return std::make_pair(load, store);
}

// Internal plan executor.
// For in-place transforms, in_buffer == out_buffer.
void TransformPowX(const ExecPlan&       execPlan,
                   void*                 in_buffer[],
                   void*                 out_buffer[],
                   rocfft_execution_info info)
{
    assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    // we can log profile information if we're on the null stream,
    // since we will be able to wait for the transform to finish
    bool            emit_profile_log  = LOG_PROFILE_ENABLED() && !info->rocfft_stream;
    bool            emit_kernelio_log = LOG_KERNELIO_ENABLED();
    rocfft_ostream* kernelio_stream   = LogSingleton::GetInstance().GetKernelIOOS();
    float           max_memory_bw     = 0.0;
    hipEvent_t      start, stop;
    if(emit_profile_log)
    {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        max_memory_bw = max_memory_bandwidth_GB_per_s();
    }

    // assign callbacks to the node that are actually doing the
    // loading and storing to/from global memory
    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = get_load_store_nodes(execPlan);

    load_node->callbacks.load_cb_fn        = info->callbacks.load_cb_fn;
    load_node->callbacks.load_cb_data      = info->callbacks.load_cb_data;
    load_node->callbacks.load_cb_lds_bytes = info->callbacks.load_cb_lds_bytes;

    store_node->callbacks.store_cb_fn        = info->callbacks.store_cb_fn;
    store_node->callbacks.store_cb_data      = info->callbacks.store_cb_data;
    store_node->callbacks.store_cb_lds_bytes = info->callbacks.store_cb_lds_bytes;

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DeviceCallIn data;
        data.node          = execPlan.execSeq[i];
        data.rocfft_stream = (info == nullptr) ? 0 : info->rocfft_stream;
        data.deviceProp    = execPlan.deviceProp;

        // Size of complex type
        const size_t complexTSize = (data.node->precision == rocfft_precision_single)
                                        ? sizeof(float) * 2
                                        : sizeof(double) * 2;

        switch(data.node->obIn)
        {
        case OB_USER_IN:
            data.bufIn[0] = in_buffer[0];
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufIn[1] = in_buffer[1];
            }
            break;
        case OB_USER_OUT:
            data.bufIn[0] = out_buffer[0];
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufIn[1] = out_buffer[1];
            }
            break;
        case OB_TEMP:
            data.bufIn[0] = info->workBuffer;
            if(data.node->inArrayType == rocfft_array_type_complex_planar
               || data.node->inArrayType == rocfft_array_type_hermitian_planar)
            {
                // Assume planar using the same extra size of memory as
                // interleaved format, and we just need to split it for
                // planar.
                data.bufIn[1]
                    = (void*)((char*)info->workBuffer + execPlan.workBufSize * complexTSize / 2);
            }
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufIn[0]
                = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufIn[0] = (void*)((char*)info->workBuffer
                                    + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                       + data.node->iOffset)
                                          * complexTSize);
            break;
        case OB_UNINIT:
            rocfft_cerr << "Error: operating buffer not initialized for kernel!\n";
            assert(data.node->obIn != OB_UNINIT);
        default:
            rocfft_cerr << "Error: operating buffer not specified for kernel!\n";
            assert(false);
        }

        switch(data.node->obOut)
        {
        case OB_USER_IN:
            data.bufOut[0] = in_buffer[0];
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufOut[1] = in_buffer[1];
            }
            break;
        case OB_USER_OUT:
            data.bufOut[0] = out_buffer[0];
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                data.bufOut[1] = out_buffer[1];
            }
            break;
        case OB_TEMP:
            data.bufOut[0] = info->workBuffer;
            if(data.node->outArrayType == rocfft_array_type_complex_planar
               || data.node->outArrayType == rocfft_array_type_hermitian_planar)
            {
                // assume planar using the same extra size of memory as
                // interleaved format, and we just need to split it for
                // planar.
                data.bufOut[1]
                    = (void*)((char*)info->workBuffer + execPlan.workBufSize * complexTSize / 2);
            }
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufOut[0]
                = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufOut[0] = (void*)((char*)info->workBuffer
                                     + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                        + data.node->oOffset)
                                           * complexTSize);
            break;
        default:
            assert(false);
        }

        data.gridParam = execPlan.gridParam[i];

        if(emit_kernelio_log)
        {
            *kernelio_stream << "--- --- kernel " << i << " (" << PrintScheme(data.node->scheme)
                             << ") input:" << std::endl;

            hipDeviceSynchronize();
            DebugPrintBuffer(*kernelio_stream,
                             data.node->inArrayType,
                             data.node->precision,
                             data.bufIn,
                             data.node->length,
                             data.node->inStride,
                             data.node->iDist,
                             data.node->iOffset,
                             data.node->batch);
        }

        DevFnCall fn = execPlan.devFnCall[i];
        if(fn)
        {
#ifdef REF_DEBUG
            rocfft_cout << "\n---------------------------------------------\n";
            rocfft_cout << "\n\nkernel: " << i << std::endl;
            rocfft_cout << "\tscheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
            rocfft_cout << "\titype: " << execPlan.execSeq[i]->inArrayType << std::endl;
            rocfft_cout << "\totype: " << execPlan.execSeq[i]->outArrayType << std::endl;
            rocfft_cout << "\tlength: ";
            for(const auto& i : execPlan.execSeq[i]->length)
            {
                rocfft_cout << i << " ";
            }
            rocfft_cout << std::endl;
            rocfft_cout << "\tbatch:   " << execPlan.execSeq[i]->batch << std::endl;
            rocfft_cout << "\tidist:   " << execPlan.execSeq[i]->iDist << std::endl;
            rocfft_cout << "\todist:   " << execPlan.execSeq[i]->oDist << std::endl;
            rocfft_cout << "\tistride:";
            for(const auto& i : execPlan.execSeq[i]->inStride)
            {
                rocfft_cout << " " << i;
            }
            rocfft_cout << std::endl;
            rocfft_cout << "\tostride:";
            for(const auto& i : execPlan.execSeq[i]->outStride)
            {
                rocfft_cout << " " << i;
            }
            rocfft_cout << std::endl;

            RefLibOp refLibOp(&data);
#endif

            // execution kernel:
            if(emit_profile_log)
                hipEventRecord(start);
            DeviceCallOut back;

            // give callback parameters to kernel launcher
            data.callbacks = execPlan.execSeq[i]->callbacks;

            // skip apply callback kernel if there's no callback
            if(data.node->scheme != CS_KERNEL_APPLY_CALLBACK
               || data.get_callback_type() != CallbackType::NONE)
            {
                fn(&data, &back);
            }
            if(emit_profile_log)
                hipEventRecord(stop);

            // If we were on the null stream, measure elapsed time
            // and emit profile logging.  If a stream was given, we
            // can't wait for the transform to finish, so we can't
            // emit any information.
            if(emit_profile_log)
            {
                hipEventSynchronize(stop);
                size_t in_size_bytes = data_size_bytes(
                    data.node->length, data.node->precision, data.node->inArrayType);
                size_t out_size_bytes = data_size_bytes(
                    data.node->length, data.node->precision, data.node->outArrayType);
                size_t total_size_bytes = (in_size_bytes + out_size_bytes) * data.node->batch;

                float duration_ms = 0.0f;
                hipEventElapsedTime(&duration_ms, start, stop);
                auto exec_bw        = execution_bandwidth_GB_per_s(total_size_bytes, duration_ms);
                auto efficiency_pct = 0.0;
                if(max_memory_bw != 0.0)
                    efficiency_pct = 100.0 * exec_bw / max_memory_bw;
                log_profile(__func__,
                            "scheme",
                            PrintScheme(execPlan.execSeq[i]->scheme),
                            "duration_ms",
                            duration_ms,
                            "in_size",
                            std::make_pair(static_cast<const size_t*>(data.node->length.data()),
                                           data.node->length.size()),
                            "total_size_bytes",
                            total_size_bytes,
                            "exec_GB_s",
                            exec_bw,
                            "max_mem_GB_s",
                            max_memory_bw,
                            "bw_efficiency_pct",
                            efficiency_pct);
            }

#ifdef REF_DEBUG
            refLibOp.VerifyResult(&data);
#endif
        }
        else
        {
            rocfft_cout << "null ptr function call error\n";
        }

        if(emit_kernelio_log)
        {
            hipError_t err = hipPeekAtLastError();
            if(err != hipSuccess)
            {
                *kernelio_stream << "Error: " << hipGetErrorName(err) << ", "
                                 << hipGetErrorString(err) << std::endl;
                exit(-1);
            }
            hipDeviceSynchronize();
            *kernelio_stream << "executed kernel " << i << " (" << PrintScheme(data.node->scheme)
                             << ")" << std::endl;
        }
    }

    if(emit_kernelio_log)
    {
        *kernelio_stream << "final output:\n";
        DebugPrintBuffer(*kernelio_stream,
                         execPlan.rootPlan->outArrayType,
                         execPlan.rootPlan->precision,
                         out_buffer,
                         execPlan.rootPlan->length,
                         execPlan.rootPlan->outStride,
                         execPlan.rootPlan->oDist,
                         execPlan.rootPlan->oOffset,
                         execPlan.rootPlan->batch);
        *kernelio_stream << std::endl;
    }
    if(emit_profile_log)
    {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
}
