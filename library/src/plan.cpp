// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "plan.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "../../shared/precision_type.h"
#include "../../shared/ptrdiff.h"
#include "assignment_policy.h"
#include "enum_printer.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "node_factory.h"
#include "rocfft/rocfft-version.h"
#include "rocfft/rocfft.h"
#include "rocfft_ostream.hpp"
#include "rtc_kernel.h"
#include "solution_map.h"
#include "tuning_helper.h"
#include "tuning_plan_tuner.h"

#include <algorithm>
#include <assert.h>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

// clang-format off
#define ROCFFT_VERSION_STRING (TO_STR(rocfft_version_major) "." \
                               TO_STR(rocfft_version_minor) "." \
                               TO_STR(rocfft_version_patch) "." \
                               TO_STR(rocfft_version_tweak) )
// clang-format on

rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description,
                                                       const double            scale_factor)
{
    log_trace(__func__, "description", description, "scale", scale_factor);
    if(!std::isfinite(scale_factor))
        return rocfft_status_invalid_arg_value;
    description->storeOps.scale_factor = scale_factor;
    return rocfft_status_success;
}

static size_t offset_count(rocfft_array_type type)
{
    // planar data has 2 sets of offsets, otherwise we have one
    return type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar
               ? 2
               : 1;
}

void rocfft_plan_description_t::init_defaults(rocfft_transform_type      transformType,
                                              rocfft_result_placement    placement,
                                              const std::vector<size_t>& lengths,
                                              const std::vector<size_t>& outputLengths)
{
    const size_t rank = lengths.size();

    // assume interleaved data
    if(inArrayType == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            inArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            inArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            inArrayType = rocfft_array_type_real;
            break;
        }
    }
    if(outArrayType == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            outArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            outArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            outArrayType = rocfft_array_type_real;
            break;
        }
    }

    // Set inStrides, if not specified
    if(inStrides.empty())
    {
        inStrides.push_back(1);

        if((transformType == rocfft_transform_type_real_forward)
           && (placement == rocfft_placement_inplace))
        {
            // real-to-complex in-place
            size_t dist = 2 * outputLengths[0];

            for(size_t i = 1; i < rank; i++)
            {
                inStrides.push_back(dist);
                dist *= lengths[i];
            }
        }
        else
        {
            // Set the inStrides to deal with contiguous data
            for(size_t i = 1; i < rank; i++)
                inStrides.push_back(lengths[i - 1] * inStrides[i - 1]);
        }
    }

    // Set outStrides, if not specified
    if(outStrides.empty())
    {
        outStrides.push_back(1);

        if((transformType == rocfft_transform_type_real_inverse)
           && (placement == rocfft_placement_inplace))
        {
            // complex-to-real in-place
            size_t dist = 2 * lengths[0];

            for(size_t i = 1; i < rank; i++)
            {
                outStrides.push_back(dist);
                dist *= lengths[i];
            }
        }
        else
        {
            // Set the outStrides to deal with contiguous data
            for(size_t i = 1; i < rank; i++)
                outStrides.push_back(lengths[i - 1] * outStrides[i - 1]);
        }
    }

    // Set in and out Distances, if not specified
    if(inDist == 0)
    {
        inDist = lengths[rank - 1] * inStrides[rank - 1];
    }
    if(outDist == 0)
    {
        outDist = lengths[rank - 1] * outStrides[rank - 1];
    }
}

void rocfft_plan_t::sort()
{
    // copy the lengths + strides separately, and then sort them
    // fastest to slowest.
    struct rocfft_iodim
    {
        size_t length;
        size_t olength;
        size_t istride;
        size_t ostride;
    };

    // complex-complex transforms can be freely reordered starting from
    // the fastest dimension.  real-complex has to leave the fastest
    // dimension alone
    const size_t start_dim = (transformType == rocfft_transform_type_complex_forward
                              || transformType == rocfft_transform_type_complex_inverse)
                                 ? 0
                                 : 1;

    std::vector<rocfft_iodim> iodims;
    for(size_t dim = start_dim; dim < rank; ++dim)
        iodims.push_back(rocfft_iodim{
            lengths[dim], outputLengths[dim], desc.inStrides[dim], desc.outStrides[dim]});
    if(iodims.empty())
        return;

    bool sort_on_istride = true;
    auto sorter          = [sort_on_istride](const rocfft_iodim& a, const rocfft_iodim& b) {
        // move any lengths of 1 to the end
        if(a.length == 1 && b.length != 1)
            return false;
        if(b.length == 1 && a.length != 1)
            return true;
        return sort_on_istride ? (a.istride < b.istride) : (a.ostride < b.ostride);
    };

    // sort on istride first
    std::sort(iodims.begin(), iodims.end(), sorter);

    // if that means ostride is no longer sorted, then don't bother
    // changing anything - the user is asking for some kind of
    // transposed FFT so let's just assume they know what they're doing
    sort_on_istride = false;
    if(!std::is_sorted(iodims.begin(), iodims.end(), sorter))
        return;

    // chop off any lengths of 1 from the end
    while(iodims.size() > 1 && iodims.back().length == 1)
    {
        --rank;
        iodims.pop_back();
    }
    // copy back the sorted lengths + strides
    for(size_t dim = start_dim; dim < rank; ++dim)
    {
        lengths[dim]         = iodims[dim - start_dim].length;
        outputLengths[dim]   = iodims[dim - start_dim].olength;
        desc.inStrides[dim]  = iodims[dim - start_dim].istride;
        desc.outStrides[dim] = iodims[dim - start_dim].ostride;
    }
}

bool rocfft_plan_t::is_contiguous(const std::vector<size_t>& length,
                                  const std::vector<size_t>& stride,
                                  size_t                     dist)
{
    size_t expected_stride = 1;
    auto   stride_it       = stride.begin();
    auto   length_it       = length.begin();
    for(; stride_it != stride.end() && length_it != length.end(); ++stride_it, ++length_it)
    {
        if(*stride_it != expected_stride)
            return false;
        expected_stride *= *length_it;
    }
    return expected_stride == dist;
}

bool rocfft_plan_t::is_contiguous_input()
{
    return is_contiguous(lengths, desc.inStrides, desc.inDist);
}
bool rocfft_plan_t::is_contiguous_output()
{
    return is_contiguous(lengths, desc.outStrides, desc.outDist);
}

size_t rocfft_plan_t::AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                                       const std::vector<size_t>&       antecedents)
{
    // ensure antecedents all exist
    if(std::any_of(
           antecedents.begin(), antecedents.end(), [=](size_t i) { return i >= multiPlan.size(); }))
        throw std::runtime_error("antecedent does not exist");

    multiPlan.emplace_back(std::move(item));
    multiPlanAdjacency.emplace_back(antecedents);
    // return index of new item
    return multiPlan.size() - 1;
}

void rocfft_plan_t::AddAntecedent(size_t itemIdx, size_t antecedentIdx)
{
    // we're not implementing full dependency cycle checks but at least
    // we can check for obvious errors
    if(itemIdx >= multiPlan.size() || antecedentIdx >= multiPlan.size() || itemIdx == antecedentIdx)
        throw std::runtime_error("invalid antecedent during plan creation");

    auto& antecedents = multiPlanAdjacency[itemIdx];
    if(std::find(antecedents.begin(), antecedents.end(), antecedentIdx) == antecedents.end())
        antecedents.push_back(antecedentIdx);
}

size_t rocfft_plan_t::WorkBufBytes() const
{
    auto base_type_size = real_type_size(precision);

    // Something wants to know how much work buffer to allocate, but
    // our plan could span multiple ranks and devices.  For now, just
    // get the largest work buf size of any item in the
    // multi-rank/device plan.
    size_t workBufBytes = 0;
    for(const auto& i : multiPlan)
    {
        if(i)
            workBufBytes = std::max(workBufBytes, i->WorkBufBytes(base_type_size));
    }
    return workBufBytes;
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type,
                                                      const size_t*           in_offsets,
                                                      const size_t*           out_offsets,
                                                      const size_t            in_strides_size,
                                                      const size_t*           in_strides,
                                                      const size_t            in_distance,
                                                      const size_t            out_strides_size,
                                                      const size_t*           out_strides,
                                                      const size_t            out_distance)
{
    log_trace(__func__,
              "description",
              description,
              "in_array_type",
              in_array_type,
              "out_array_type",
              out_array_type,
              "in_offsets",
              std::make_pair(in_offsets, offset_count(in_array_type)),
              "out_offsets",
              std::make_pair(out_offsets, offset_count(out_array_type)),
              "in_strides",
              std::make_pair(in_strides, in_strides_size),
              "in_distance",
              in_distance,
              "out_strides",
              std::make_pair(out_strides, out_strides_size),
              "out_distance",
              out_distance);

    description->inArrayType  = in_array_type;
    description->outArrayType = out_array_type;

    if(in_offsets != nullptr)
    {
        description->inOffset[0] = in_offsets[0];
        if((in_array_type == rocfft_array_type_complex_planar)
           || (in_array_type == rocfft_array_type_hermitian_planar))
            description->inOffset[1] = in_offsets[1];
    }

    if(out_offsets != nullptr)
    {
        description->outOffset[0] = out_offsets[0];
        if((out_array_type == rocfft_array_type_complex_planar)
           || (out_array_type == rocfft_array_type_hermitian_planar))
            description->outOffset[1] = out_offsets[1];
    }

    if(in_strides != nullptr)
    {
        std::copy(
            in_strides, in_strides + in_strides_size, std::back_inserter(description->inStrides));
    }

    if(in_distance != 0)
        description->inDist = in_distance;

    if(out_strides != nullptr)
    {
        std::copy(out_strides,
                  out_strides + out_strides_size,
                  std::back_inserter(description->outStrides));
    }

    if(out_distance != 0)
        description->outDist = out_distance;

    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_create(rocfft_plan_description* description)
{
    rocfft_plan_description desc = new rocfft_plan_description_t;
    *description                 = desc;
    log_trace(__func__, "description", *description);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description)
{
    log_trace(__func__, "description", description);
    if(description != nullptr)
        delete description;
    return rocfft_status_success;
}

rocfft_status rocfft_field_create(rocfft_field* field)
{
    *field = new rocfft_field_t;
    log_trace(__func__, "field", *field);
    return rocfft_status_success;
}

rocfft_status rocfft_field_destroy(rocfft_field field)
{
    log_trace(__func__, "field", field);
    delete field;
    return rocfft_status_success;
}

bool rocfft_brick_t::empty() const
{
    auto len = length();
    return std::any_of(len.begin(), len.end(), [](int l) { return l == 0; });
}

size_t rocfft_brick_t::count_elems() const
{
    auto len = length();
    return product(len.begin(), len.end());
}

std::vector<size_t> rocfft_brick_t::contiguous_strides() const
{
    std::vector<size_t> ret;
    size_t              dist = 1;

    auto len = length();
    for(size_t i = 0; i < len.size(); ++i)
    {
        ret.push_back(dist);
        dist *= len[i];
    }
    return ret;
}

bool rocfft_brick_t::is_contiguous() const
{
    auto contiguous = contiguous_strides();

    auto len = length();
    // only care about dimensions whose length is greater than 1
    for(size_t i = 1; i < len.size(); ++i)
    {
        if(len[i] > 1)
        {
            if(contiguous[i] != stride[i])
                return false;
        }
    }
    return true;
}

bool rocfft_brick_t::is_contiguous_in_field(const std::vector<size_t>& field_length,
                                            const std::vector<size_t>& field_stride) const
{
    const auto brick_len = length();

    // a contiguous brick in a field is shorter than field length
    // only on the highest dimension, ignoring dimensions of
    // length-1, and strides must match the field for all those
    // dimensions.
    size_t shortDim       = std::numeric_limits<size_t>::max();
    size_t highestNot1Dim = std::numeric_limits<size_t>::max();
    for(size_t i = 0; i < brick_len.size(); ++i)
    {
        if(field_length[i] == 1)
            continue;

        highestNot1Dim = i;
        if(brick_len[i] < field_length[i])
        {
            // another dim is already short?  this isn't contiguous
            if(shortDim != std::numeric_limits<size_t>::max())
                return false;
            shortDim = i;
        }
    }

    return shortDim == highestNot1Dim;
}

rocfft_brick_t rocfft_brick_t::intersect(const rocfft_brick_t& other) const
{
    rocfft_brick_t ret;

    for(size_t i = 0; i < lower.size(); ++i)
        ret.lower.push_back(std::max(lower[i], other.lower[i]));

    for(size_t i = 0; i < upper.size(); ++i)
        ret.upper.push_back(std::min(upper[i], other.upper[i]));

    return ret;
}

bool rocfft_brick_t::equal_coords(const rocfft_brick_t& other) const
{
    return this->lower == other.lower && this->upper == other.upper;
}

size_t rocfft_brick_t::offset_in_field(const std::vector<size_t>& fieldStride) const
{
    return std::inner_product(lower.begin(), lower.end(), fieldStride.begin(), 0);
}

std::string rocfft_brick_t::str() const
{
    std::string ret;
    ret += "lower ";
    for(auto i : lower)
    {
        ret += " ";
        ret += std::to_string(i);
    }

    ret += " upper ";
    for(auto i : upper)
    {
        ret += " ";
        ret += std::to_string(i);
    }

    ret += " stride ";
    for(auto i : stride)
    {
        ret += " ";
        ret += std::to_string(i);
    }
    ret += " device ";
    ret += std::to_string(device);
    return ret;
}

rocfft_status rocfft_field_add_brick(rocfft_field field, rocfft_brick brick)
{
    log_trace(__func__, "field", field, "brick", brick);
    if(!field || !brick)
        return rocfft_status_invalid_arg_value;
    field->bricks.emplace_back(*brick);
    return rocfft_status_success;
}

rocfft_status rocfft_brick_create(rocfft_brick* brick,
                                  const size_t* field_lower,
                                  const size_t* field_upper,
                                  const size_t* brick_stride,
                                  size_t        dim,
                                  int           deviceID)
{
    log_trace(__func__,
              "brick",
              brick,
              "field_lower",
              field_lower,
              "field_upper",
              field_upper,
              "brick_stride",
              brick_stride,
              "dim",
              dim,
              "deviceID",
              deviceID);
    if(!brick)
        return rocfft_status_invalid_arg_value;

    auto brick_ptr = std::make_unique<rocfft_brick_t>();
    std::copy_n(field_lower, dim, std::back_inserter(brick_ptr->lower));
    std::copy_n(field_upper, dim, std::back_inserter(brick_ptr->upper));
    std::copy_n(brick_stride, dim, std::back_inserter(brick_ptr->stride));

    brick_ptr->device = deviceID;
    *brick            = brick_ptr.release();
    return rocfft_status_success;
}

rocfft_status rocfft_brick_destroy(rocfft_brick brick)
{
    log_trace(__func__, "brick", brick);
    delete brick;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_add_infield(rocfft_plan_description description,
                                                  rocfft_field            field)
{
    log_trace(__func__, "description", description, "field", field);
    if(!description || !field || field->bricks.empty())
        return rocfft_status_invalid_arg_value;
    description->inFields.push_back(*field);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_add_outfield(rocfft_plan_description description,
                                                   rocfft_field            field)
{
    log_trace(__func__, "description", description, "field", field);
    if(!description || !field || field->bricks.empty())
        return rocfft_status_invalid_arg_value;
    description->outFields.push_back(*field);
    return rocfft_status_success;
}

std::string rocfft_bench_command(rocfft_plan plan)
{
    std::stringstream bench;
    bench << "rocfft-bench --length ";
    std::ostream_iterator<size_t> bench_iter(bench, " ");
    std::copy(plan->lengths.rbegin(), plan->lengths.rend(), bench_iter);
    bench << "-b " << plan->batch << " ";

    if(plan->placement == rocfft_placement_notinplace)
        bench << "-o ";

    bench << "-t " << plan->transformType << " ";

    bench << "--precision ";
    bench << precision_name(plan->precision) << " ";
    bench << "--itype " << plan->desc.inArrayType << " ";
    bench << "--otype " << plan->desc.outArrayType << " ";
    bench << "--istride ";
    std::copy(plan->desc.inStrides.rbegin(), plan->desc.inStrides.rend(), bench_iter);
    bench << "--ostride ";
    std::copy(plan->desc.outStrides.rbegin(), plan->desc.outStrides.rend(), bench_iter);
    bench << "--idist " << plan->desc.inDist << " ";
    bench << "--odist " << plan->desc.outDist << " ";
    bench << "--ioffset ";
    std::copy(plan->desc.inOffset.begin(), plan->desc.inOffset.end(), bench_iter);
    bench << "--ooffset ";
    std::copy(plan->desc.outOffset.begin(), plan->desc.outOffset.end(), bench_iter);
    return bench.str();
}

void set_bluestein_strides(const rocfft_plan plan, NodeMetaData& planData)
{
    std::array<size_t, 3> inStridesBlue  = {0, 0, 0};
    std::array<size_t, 3> outStridesBlue = {0, 0, 0};
    std::array<size_t, 3> lengthsBlue    = {0, 0, 0};
    size_t                inDistBlue     = 0;
    size_t                outDistBlue    = 0;

    const auto precision     = plan->precision;
    const auto transformType = plan->transformType;
    const auto rank          = plan->rank;
    const auto lengths       = plan->lengths;
    const auto placement     = plan->placement;
    const auto dimension     = planData.dimension;

    assert(rank == dimension);

    // for real inverse transforms we need to look at the complex length
    auto fftLength
        = transformType == rocfft_transform_type_real_inverse ? plan->outputLengths : plan->lengths;

    lengthsBlue[0] = NodeFactory::SupportedLength(precision, fftLength[0])
                         ? fftLength[0]
                         : NodeFactory::GetBluesteinLength(precision, fftLength[0]);
    for(size_t i = 1; i < dimension; i++)
        lengthsBlue[i] = fftLength[i];

    // =================================
    // inStrides
    // =================================
    inStridesBlue[0] = 1;

    if((transformType == rocfft_transform_type_real_forward)
       && (placement == rocfft_placement_inplace))
    {
        // real-to-complex in-place
        size_t dist = 2 * (1 + (lengthsBlue[0]) / 2);

        for(size_t i = 1; i < rank; i++)
        {
            inStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        inDistBlue = dist;
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        // complex-to-real
        size_t dist = 1 + (lengthsBlue[0]) / 2;

        for(size_t i = 1; i < rank; i++)
        {
            inStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        inDistBlue = dist;
    }
    else
    {
        // Set the inStrides to deal with contiguous data
        for(size_t i = 1; i < rank; i++)
            inStridesBlue[i] = lengthsBlue[i - 1] * inStridesBlue[i - 1];

        inDistBlue = lengthsBlue[rank - 1] * inStridesBlue[rank - 1];
    }

    // =================================
    // outStrides
    // =================================
    outStridesBlue[0] = 1;

    if((transformType == rocfft_transform_type_real_forward)
       && (placement == rocfft_placement_inplace))
    {
        // real-to-complex in-place
        size_t dist = 2 * (1 + (lengthsBlue[0]) / 2);

        for(size_t i = 1; i < rank; i++)
        {
            outStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        outDistBlue = dist;
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        // complex-to-real
        size_t dist = 1 + (lengthsBlue[0]) / 2;

        for(size_t i = 1; i < rank; i++)
        {
            outStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        outDistBlue = dist;
    }
    else
    {
        // Set the inStrides to deal with contiguous data
        for(size_t i = 1; i < rank; i++)
            outStridesBlue[i] = lengthsBlue[i - 1] * outStridesBlue[i - 1];

        outDistBlue = lengthsBlue[rank - 1] * outStridesBlue[rank - 1];
    }

    for(size_t i = 0; i < dimension; i++)
    {
        planData.inStrideBlue.push_back(inStridesBlue[i]);
        planData.outStrideBlue.push_back(outStridesBlue[i]);
    }
    planData.iDistBlue = inDistBlue;
    planData.oDistBlue = outDistBlue;
}

// return an ExecPlan that transposes a brick
std::unique_ptr<ExecPlan> transpose_brick(int                        deviceID,
                                          const std::vector<size_t>& length,
                                          rocfft_precision           precision,
                                          rocfft_array_type          arrayType,
                                          BufferPtr                  inputPtr,
                                          size_t                     offsetIn,
                                          const std::vector<size_t>& strideIn,
                                          BufferPtr                  outputPtr,
                                          size_t                     offsetOut,
                                          const std::vector<size_t>& strideOut,
                                          std::string&&              description)
{
    rocfft_scoped_device dev(deviceID);

    auto      execPlanMultiItem = std::make_unique<ExecPlan>();
    ExecPlan& execPlan          = *execPlanMultiItem;

    execPlan.deviceID   = deviceID;
    execPlan.deviceProp = get_curr_device_prop();
    execPlan.mgpuPlan   = true;

    // add input buffers provided by users
    execPlan.inputPtr  = inputPtr;
    execPlan.outputPtr = outputPtr;

    // transpose 2D
    switch(length.size())
    {
    case 2:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, nullptr);

        execPlan.rootPlan->deviceProp = execPlan.deviceProp;

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 2;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
    case 3:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, nullptr);

        execPlan.rootPlan->deviceProp = execPlan.deviceProp;

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 3;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
        // 4D is required if we have a 3D problem + batch
    case 4:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, nullptr);

        execPlan.rootPlan->deviceProp = execPlan.deviceProp;

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 4;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
    default:
        throw std::runtime_error("unsupported transpose_brick dimension");
    }

    // Set input/output buffers - these will either be actual user
    // input/output (when packing/unpacking for communication), or we
    // want the kernel to use overridden pointers that we allocate
    // during plan creation, which are also passed to look like user
    // input/output pointers.
    execPlan.rootPlan->obIn  = OB_USER_IN;
    execPlan.rootPlan->obOut = OB_USER_OUT;

    execPlan.oLength            = execPlan.rootPlan->length;
    execPlan.rootPlan->inStride = strideIn;

    execPlan.rootPlan->precision = precision;
    execPlan.rootPlan->placement = rocfft_placement_notinplace;
    execPlan.rootPlan->iOffset   = offsetIn;
    execPlan.rootPlan->oOffset   = offsetOut;

    execPlan.rootPlan->inArrayType  = arrayType;
    execPlan.rootPlan->outArrayType = arrayType;

    execPlan.execSeq.push_back(execPlan.rootPlan.get());
    execPlan.rootPlan->CreateDevKernelArgs();

    execPlan.rootPlan->comments.emplace_back(std::move(description));

    // FIXME: on multi-rank plans, we should only compile for the
    // current rank
    RuntimeCompilePlan(execPlan);

    // grid params are set during runtime compilation, put them on
    // the execPlan so they're known at exec time
    execPlan.devFnCall.push_back(nullptr);
    auto& gp       = execPlan.gridParam.emplace_back();
    dim3  gridDim  = execPlan.execSeq.front()->compiledKernel.get()->gridDim;
    dim3  blockDim = execPlan.execSeq.front()->compiledKernel.get()->blockDim;
    gp.b_x         = gridDim.x;
    gp.b_y         = gridDim.y;
    gp.b_z         = gridDim.z;
    gp.wgs_x       = blockDim.x;
    gp.wgs_y       = blockDim.y;
    gp.wgs_z       = blockDim.z;

    return execPlanMultiItem;
}

// RAII struct to 'lease' a temp buffer from a multimap of per-device
// buffers.  When this struct is destroyed, the buffer is returned to
// the map for reuse.
struct TempBufferLease
{
    TempBufferLease(std::multimap<int, gpubuf>& _tempBuffers,
                    int                         _deviceID,
                    size_t                      _elems,
                    size_t                      _elem_size)
        : deviceID(_deviceID)
        , tempBuffers(&_tempBuffers)
    {
        // return an existing buffer that's big enough, if one exists
        const size_t alloc_size = _elems * _elem_size;
        for(auto i = tempBuffers->lower_bound(deviceID); i != tempBuffers->upper_bound(deviceID);
            ++i)
        {
            if(i->second.size() >= alloc_size)
            {
                // leasing out the temp buffer, remove it from the map
                buf = std::move(i->second);
                tempBuffers->erase(i);
                return;
            }
        }
        // no buffer was found, allocate a new one
        rocfft_scoped_device dev(deviceID);
        if(buf.alloc(alloc_size) != hipSuccess)
            throw std::runtime_error("hipMalloc failure");

        if(LOG_PLAN_ENABLED())
            *LogSingleton::GetInstance().GetPlanOS()
                << "temp buffer " << buf.data() << ", device " << deviceID << ", elems " << _elems
                << ", size " << _elem_size << std::endl;
    }
    ~TempBufferLease()
    {
        // return the buffer to the map
        tempBuffers->emplace(std::make_pair(deviceID, std::move(buf)));
    }
    // allow moves, disallow copies
    TempBufferLease(TempBufferLease&& other)
        : deviceID(other.deviceID)
        , tempBuffers(other.tempBuffers)
        , buf(std::move(other.buf))
    {
    }
    TempBufferLease& operator=(TempBufferLease&& other)
    {
        deviceID    = other.deviceID;
        tempBuffers = other.tempBuffers;
        buf         = std::move(other.buf);
        return *this;
    }
    TempBufferLease(const TempBufferLease& other) = delete;
    TempBufferLease& operator=(const TempBufferLease& other) = delete;

    void* data()
    {
        return buf.data();
    }

private:
    int                         deviceID;
    std::multimap<int, gpubuf>* tempBuffers;
    gpubuf                      buf;
};

std::vector<size_t> rocfft_plan_t::GatherBricksToField(int currentDevice,
                                                       const std::vector<rocfft_brick_t>& bricks,
                                                       rocfft_precision                   precision,
                                                       rocfft_array_type                  arrayType,
                                                       const std::vector<size_t>& field_length,
                                                       const std::vector<size_t>& field_stride,
                                                       BufferPtr                  output,
                                                       const std::vector<size_t>& antecedents,
                                                       size_t                     elem_size)
{
    std::vector<size_t>            outputPlanItems;
    std::vector<TempBufferLease>   gatherPackBufs;
    std::optional<TempBufferLease> gatherDestBuf;

    // create gather operation
    auto gatherPtr       = std::make_unique<CommGather>();
    auto gather          = gatherPtr.get();
    gather->precision    = precision;
    gather->arrayType    = arrayType;
    gather->destDeviceID = currentDevice;

    // add gather to the plan first - we will add operations to it
    // later
    size_t gatherIdx = AddMultiPlanItem(std::move(gatherPtr), antecedents);

    const bool gatherToTemp
        = std::any_of(bricks.begin(), bricks.end(), [&](const rocfft_brick_t& b) {
              return !b.is_contiguous() || !b.is_contiguous_in_field(field_length, field_stride);
          });
    if(gatherToTemp)
    {
        gatherDestBuf
            = std::make_optional<TempBufferLease>(tempBuffers,
                                                  currentDevice,
                                                  product(field_length.begin(), field_length.end()),
                                                  elem_size);
        gather->destPtr = BufferPtr::temp(gatherDestBuf->data());
    }
    else
    {
        gather->destPtr = output;
    }

    // we'll be packing the brick data contiguously into the output,
    // so keep track of how much of the output we've filled up
    size_t contiguousOffset = 0;
    for(size_t brickIdx = 0; brickIdx < bricks.size(); ++brickIdx)
    {
        const auto& b = bricks[brickIdx];
        if(b.is_contiguous())
        {
            // contiguous brick, just copy the data
            gather->ops.emplace_back(
                b.device, BufferPtr::user_input(brickIdx), 0, contiguousOffset, b.count_elems());
        }
        else
        {
            // allocate temp memory for the pack
            gatherPackBufs.emplace_back(tempBuffers, b.device, b.count_elems(), elem_size);

            // brick is not contiguous, insert a pack node on the brick's device
            std::string description = "pack brick " + std::to_string(brickIdx) + " before gather";
            auto        packIdx
                = AddMultiPlanItem(transpose_brick(b.device,
                                                   b.length(),
                                                   precision,
                                                   arrayType,
                                                   BufferPtr::user_input(brickIdx),
                                                   0,
                                                   b.stride,
                                                   BufferPtr::temp(gatherPackBufs.back().data()),
                                                   contiguousOffset,
                                                   b.contiguous_strides(),
                                                   std::move(description)),
                                   antecedents);
            AddAntecedent(gatherIdx, packIdx);

            gather->ops.emplace_back(b.device,
                                     BufferPtr::temp(gatherPackBufs.back().data()),
                                     0,
                                     contiguousOffset,
                                     b.count_elems());
        }

        // unpack if necessary
        if(!b.is_contiguous() || !b.is_contiguous_in_field(field_length, field_stride))
        {
            std::string description = "unpack brick " + std::to_string(brickIdx) + " after gather";

            outputPlanItems.push_back(
                AddMultiPlanItem(transpose_brick(currentDevice,
                                                 b.length(),
                                                 precision,
                                                 arrayType,
                                                 BufferPtr::temp(gatherDestBuf->data()),
                                                 contiguousOffset,
                                                 b.contiguous_strides(),
                                                 output,
                                                 b.offset_in_field(field_stride),
                                                 field_stride,
                                                 std::move(description)),
                                 {gatherIdx}));
        }

        contiguousOffset += b.count_elems();
    }

    if(outputPlanItems.empty())
    {
        // following items in the plan just depend on the gather completing
        outputPlanItems.push_back(gatherIdx);
    }
    return outputPlanItems;
}

std::vector<size_t> rocfft_plan_t::ScatterFieldToBricks(int                        currentDevice,
                                                        BufferPtr                  input,
                                                        rocfft_precision           precision,
                                                        rocfft_array_type          arrayType,
                                                        const std::vector<size_t>& field_length,
                                                        const std::vector<size_t>& field_stride,
                                                        const std::vector<rocfft_brick_t>& bricks,
                                                        const std::vector<size_t>& antecedents,
                                                        size_t                     elem_size)
{
    std::vector<size_t>            outputPlanItems;
    std::vector<TempBufferLease>   scatterPackBufs;
    std::optional<TempBufferLease> scatterSrcBuf;

    // create scatter operation
    auto scatterPtr = std::make_unique<CommScatter>();
    auto scatter    = scatterPtr.get();

    scatter->precision   = precision;
    scatter->arrayType   = arrayType;
    scatter->srcDeviceID = currentDevice;

    // add scatter to the multi-plan first, add operations afterwards
    auto scatterIdx = AddMultiPlanItem(std::move(scatterPtr), antecedents);

    const bool scatterFromTemp
        = std::any_of(bricks.begin(), bricks.end(), [&](const rocfft_brick_t& b) {
              return !b.is_contiguous_in_field(field_length, field_stride);
          });
    if(scatterFromTemp)
    {
        scatterSrcBuf
            = std::make_optional<TempBufferLease>(tempBuffers,
                                                  currentDevice,
                                                  product(field_length.begin(), field_length.end()),
                                                  elem_size);
        scatter->srcPtr = BufferPtr::temp(scatterSrcBuf->data());
    }
    else
    {
        scatter->srcPtr = input;
    }

    // we'll be packing the brick data contiguously into the output,
    // so keep track of how much of the output we've filled up
    size_t contiguousOffset = 0;
    for(size_t brickIdx = 0; brickIdx < bricks.size(); ++brickIdx)
    {
        const auto& b = bricks[brickIdx];

        if(b.is_contiguous_in_field(field_length, field_stride))
        {
            // contiguous brick, just copy the data
            scatter->ops.emplace_back(
                b.device, BufferPtr::user_output(brickIdx), contiguousOffset, 0, b.count_elems());
        }
        else
        {
            // pack data to be contiguous
            std::string description = "pack brick " + std::to_string(brickIdx) + " before scatter";

            const auto brickLen = b.length();
            auto       packIdx  = AddMultiPlanItem(transpose_brick(currentDevice,
                                                            b.length(),
                                                            precision,
                                                            arrayType,
                                                            input,
                                                            b.offset_in_field(field_stride),
                                                            field_stride,
                                                            BufferPtr::temp(scatterSrcBuf->data()),
                                                            contiguousOffset,
                                                            b.contiguous_strides(),
                                                            std::move(description)),
                                            antecedents);
            AddAntecedent(scatterIdx, packIdx);

            // bricks are packed to be contiguous - if output is the
            // same shape, then there's no need for unpacking
            if(b.is_contiguous())
            {
                scatter->ops.emplace_back(b.device,
                                          BufferPtr::user_output(brickIdx),
                                          contiguousOffset,
                                          0,
                                          b.count_elems());
            }
            else
            {
                // allocate memory for packed data
                scatterPackBufs.emplace_back(tempBuffers, b.device, b.count_elems(), elem_size);

                // send the data
                scatter->ops.emplace_back(b.device,
                                          BufferPtr::temp(scatterPackBufs.back().data()),
                                          contiguousOffset,
                                          0,
                                          b.count_elems());

                // unpack data after sending
                description = "unpack brick " + std::to_string(brickIdx) + " after scatter";

                outputPlanItems.push_back(
                    AddMultiPlanItem(transpose_brick(b.device,
                                                     b.length(),
                                                     precision,
                                                     arrayType,
                                                     BufferPtr::temp(scatterPackBufs.back().data()),
                                                     0,
                                                     b.contiguous_strides(),
                                                     BufferPtr::user_output(brickIdx),
                                                     0,
                                                     b.stride,
                                                     std::move(description)),
                                     {scatterIdx}));
            }
        }
        contiguousOffset += b.count_elems();
    }

    // following items in the plan just depend on the scatter
    // completing if no unpacking was required
    if(outputPlanItems.empty())
    {
        outputPlanItems.push_back(scatterIdx);
    }

    return outputPlanItems;
}

void rocfft_plan_t::GatherScatterSingleDevicePlan(std::unique_ptr<ExecPlan>&& execPlanPtr)
{
    // the smart pointer will be moved into the multi-plan during this
    // function, so keep a plain non-owning pointer
    auto execPlan = execPlanPtr.get();

    // if we have no input/output fields, then the single ExecPlan is
    // exactly what we need to do
    if(desc.inFields.empty() && desc.outFields.empty())
    {
        AddMultiPlanItem(std::move(execPlanPtr), {});
        return;
    }

    // code below this line is only required for multi-device plans
    execPlan->mgpuPlan = true;

    // ensure fields are real or interleaved - planar is not supported
    if((!desc.inFields.empty() && array_type_is_planar(desc.inArrayType))
       || (!desc.outFields.empty() && array_type_is_planar(desc.outArrayType)))
    {
        throw std::runtime_error("fields must be not be planar");
    }

    const auto in_elem_size = element_size(precision, desc.inArrayType);
    // we do in-place transforms here, so allocate a buffer for the
    // complex side of real-complex transforms, since it's bigger.
    const size_t in_elem_count = compute_ptrdiff(lengths, desc.inStrides, batch, desc.inDist);

    // may need buffer(s) to do the actual FFT
    std::optional<TempBufferLease> fftBuf;
    std::optional<TempBufferLease> fftOutBuf;

    BufferPtr gatherBuf;

    // gather to temp buf if infields were specified and output is not contiguous or is also a field
    if(!desc.inFields.empty() && (!is_contiguous_output() || !desc.outFields.empty()))
    {
        fftBuf = std::make_optional<TempBufferLease>(
            tempBuffers, execPlanPtr->deviceID, in_elem_count, in_elem_size);
        gatherBuf = BufferPtr::temp(fftBuf->data());
    }
    else if(desc.inFields.empty())
        gatherBuf = BufferPtr::user_input();
    // else, we can gather directly to the output buffer
    else
        gatherBuf = BufferPtr::user_output();

    // allocate another temp buf if FFT is not-in-place and outfield was specified
    if(execPlan->rootPlan->placement == rocfft_placement_notinplace && !desc.outFields.empty())
    {
        const auto   out_elem_size  = element_size(precision, desc.outArrayType);
        const size_t out_elem_count = compute_ptrdiff(
            execPlan->rootPlan->GetOutputLength(), desc.outStrides, batch, desc.outDist);
        fftOutBuf = std::make_optional<TempBufferLease>(
            tempBuffers, execPlanPtr->deviceID, out_elem_count, out_elem_size);
    }

    std::vector<size_t> gatherIndexes;
    for(const auto& inField : desc.inFields)
    {
        // brick indexes include batch so create a set of comparable field strides
        auto fieldStrideWithBatch = desc.inStrides;
        fieldStrideWithBatch.push_back(desc.inDist);

        auto fieldLengthWithBatch = lengths;
        fieldLengthWithBatch.push_back(batch);

        auto curIndexes
            = GatherBricksToField(execPlan->deviceID,
                                  inField.bricks,
                                  execPlan->rootPlan->precision,
                                  execPlan->rootPlan->inArrayType,
                                  fieldLengthWithBatch,
                                  fieldStrideWithBatch,
                                  gatherBuf,
                                  {},
                                  element_size(precision, execPlan->rootPlan->inArrayType));
        std::copy(curIndexes.begin(), curIndexes.end(), std::back_inserter(gatherIndexes));
    }

    // data is gathered and unpacked (if necessary), run the core FFT plan we started with
    if(gatherBuf)
        execPlan->inputPtr = gatherBuf;
    else
        execPlan->inputPtr = BufferPtr::user_input();

    if(execPlan->rootPlan->placement == rocfft_placement_inplace)
        execPlan->outputPtr = execPlan->inputPtr;
    else if(fftOutBuf)
        execPlan->outputPtr = BufferPtr::temp(fftOutBuf->data());
    auto fftIdx = AddMultiPlanItem(std::move(execPlanPtr), gatherIndexes);

    // scatter data back out
    for(const auto& outField : desc.outFields)
    {
        // brick indexes include batch so create a set of comparable field strides
        auto fieldStrideWithBatch = desc.outStrides;
        fieldStrideWithBatch.push_back(desc.outDist);

        auto fieldLengthWithBatch = lengths;
        fieldLengthWithBatch.push_back(batch);

        auto scatterSrcBuf = execPlan->rootPlan->placement == rocfft_placement_notinplace
                                 ? fftOutBuf->data()
                                 : fftBuf->data();

        ScatterFieldToBricks(execPlan->deviceID,
                             BufferPtr::temp(scatterSrcBuf),
                             execPlan->rootPlan->precision,
                             execPlan->rootPlan->outArrayType,
                             fieldLengthWithBatch,
                             fieldStrideWithBatch,
                             outField.bricks,
                             {fftIdx},
                             element_size(precision, execPlan->rootPlan->outArrayType));
    }
}

// test if the specified dimension is split up across separate bricks
// in the field
static bool DimensionSplitInField(size_t length, size_t dimIdx, const rocfft_field_t& field)
{
    for(const auto& b : field.bricks)
        if(b.length()[dimIdx] != length)
            return true;
    return false;
}

// Construct a single-device execPlan - fill out the provided
// execPlan with nodes to implement the FFT.
static std::unique_ptr<ExecPlan> BuildSingleDevicePlan(NodeMetaData&         rootPlanData,
                                                       int                   deviceID,
                                                       rocfft_transform_type transformType,
                                                       LoadOps&              loadOps,
                                                       StoreOps&             storeOps)
{
    auto      execPlanMultiItem = std::make_unique<ExecPlan>();
    ExecPlan& execPlan          = *execPlanMultiItem;
    try
    {
        execPlan.deviceID   = deviceID;
        execPlan.deviceProp = rootPlanData.deviceProp;

        execPlan.rootPlan = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);

        // FIXME: some solutions require the problems to be unit_stride, otherwise the
        //   scheme-tree may not be applicable. In this case, we can't apply the solutions.
        //   Currently, it happens on Real3DEven with REAL_2D_SINGLE kernels. This needs to
        //   be detected.

        // If we are doing tuning initialzing now, we shouldn't apply any solution,
        // since we are trying enumerating solutions now
        if(TuningBenchmarker::GetSingleton().IsInitializingTuning() == false)
        {
            execPlan.rootScheme = ApplySolution(execPlan);
            if(execPlan.rootScheme)
            {
                execPlan.rootPlan = nullptr;
                execPlan.rootPlan = NodeFactory::CreateExplicitNode(
                    rootPlanData, nullptr, execPlan.rootScheme->curScheme);
            }
        }

        execPlan.iLength = rootPlanData.length;
        execPlan.oLength
            = rootPlanData.outputLength.empty() ? rootPlanData.length : rootPlanData.outputLength;

        // setup isUnitStride values
        execPlan.rootPlan->inStrideUnit  = BufferIsUnitStride(execPlan, OB_USER_IN);
        execPlan.rootPlan->outStrideUnit = BufferIsUnitStride(execPlan, OB_USER_OUT);

        // set load/store ops on the root plan
        if(loadOps.enabled())
            execPlan.rootPlan->loadOps = loadOps;
        if(storeOps.enabled())
            execPlan.rootPlan->storeOps = storeOps;

        // check if we are doing tuning init now. If yes, we just return
        // since we are not going to do the execution
        if(TuningBenchmarker::GetSingleton().IsInitializingTuning())
        {
            EnumerateTrees(execPlan);
            TuningBenchmarker::GetSingleton().GetPacket()->init_step = false;
            TuningBenchmarker::GetSingleton().GetPacket()->is_tuning = true;
            return execPlanMultiItem;
        }

        ProcessNode(execPlan); // TODO: more descriptions are needed

        // plan is compiled, no need to alloc twiddles + kargs etc
        if(rocfft_getenv("ROCFFT_INTERNAL_COMPILE_ONLY") == "1")
            return execPlanMultiItem;

        if(!PlanPowX(execPlan)) // PlanPowX enqueues the GPU kernels by function
        {
            throw std::runtime_error("Unable to create execution plan.");
        }

        // when running each solution during tuning, get the information to packet,
        // then we can dump the information to a table for analysis
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning())
        {
            if(!GetTuningKernelInfo(execPlan))
                throw std::runtime_error("Unable to get the solution info.");
        }

        return execPlanMultiItem;
    }
    catch(std::exception&)
    {
        if(LOG_PLAN_ENABLED())
            PrintNode(*LogSingleton::GetInstance().GetPlanOS(), execPlan);
        throw;
    }
}

// note: lengths and stride include batch dimension
static size_t C2COneDimension(rocfft_plan_t&             plan,
                              size_t                     dimIdx,
                              int                        deviceID,
                              const std::vector<size_t>& lengths,
                              const std::vector<size_t>& stride,
                              BufferPtr                  input,
                              BufferPtr                  output,
                              const std::vector<size_t>& antecedents)
{
    rocfft_scoped_device dev(deviceID);

    auto transformLengths = lengths;
    auto transformStride  = stride;

    // move the dimension-we-want-to-transform to the front
    std::swap(transformLengths.front(), transformLengths[dimIdx]);
    std::swap(transformStride.front(), transformStride[dimIdx]);

    NodeMetaData rootPlanData(nullptr);

    rootPlanData.batch = transformLengths.back();
    rootPlanData.iDist = transformStride.back();
    rootPlanData.oDist = transformStride.back();
    transformLengths.pop_back();
    transformStride.pop_back();

    rootPlanData.dimension = 1;
    rootPlanData.length    = transformLengths;
    rootPlanData.inStride  = transformStride;
    rootPlanData.outStride = transformStride;
    rootPlanData.direction = plan.transformType == rocfft_transform_type_complex_forward
                                     || plan.transformType == rocfft_transform_type_real_forward
                                 ? -1
                                 : 1;
    rootPlanData.placement
        = input == output ? rocfft_placement_inplace : rocfft_placement_notinplace;
    rootPlanData.precision    = plan.precision;
    rootPlanData.inArrayType  = rocfft_array_type_complex_interleaved;
    rootPlanData.outArrayType = rocfft_array_type_complex_interleaved;
    rootPlanData.deviceProp   = get_curr_device_prop();

    auto singlePlan = BuildSingleDevicePlan(
        rootPlanData, deviceID, plan.transformType, plan.desc.loadOps, plan.desc.storeOps);
    singlePlan->mgpuPlan  = true;
    singlePlan->inputPtr  = input;
    singlePlan->outputPtr = output;
    return plan.AddMultiPlanItem(std::move(singlePlan), antecedents);
}

bool rocfft_plan_t::BuildOptMultiDevicePlan()
{
    // currently, can only optimize c2c
    if(transformType != rocfft_transform_type_complex_forward
       && transformType != rocfft_transform_type_complex_inverse)
        return false;

    // must be out-of-place so that we don't have to worry about
    // overwriting an input before everything's done reading
    if(placement == rocfft_placement_inplace)
        return false;

    if(desc.inFields.empty() || desc.outFields.empty())
        return false;

    // work out what FFT dimensions are already contiguous in the input field
    std::vector<size_t> contiguousDims;
    size_t              nonContiguousDim = std::numeric_limits<size_t>::max();
    for(size_t dimIdx = 0; dimIdx < rank; ++dimIdx)
    {
        if(DimensionSplitInField(lengths[dimIdx], dimIdx, desc.inFields.front()))
            nonContiguousDim = dimIdx;
        else
            contiguousDims.push_back(dimIdx);
    }

    // can optimize if all but one FFT dim is already contiguous, and the
    // remaining dim is contiguous in output
    if(contiguousDims.size() != rank - 1
       || DimensionSplitInField(
           lengths[nonContiguousDim], nonContiguousDim, desc.outFields.front()))
        return false;

    const auto in_elem_size = element_size(precision, desc.inArrayType);
    // we do in-place transforms here, so allocate a buffer for the
    // complex side of real-complex transforms, since it's bigger.
    const size_t in_elem_count = compute_ptrdiff(lengths, desc.inStrides, batch, desc.inDist);

    std::vector<size_t> unpackIndexes;
    for(size_t inBrickIdx = 0; inBrickIdx < desc.inFields.front().bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = desc.inFields.front().bricks[inBrickIdx];

        // assume input cannot be modified, transform into temp buffer
        TempBufferLease temp(tempBuffers, inBrick.device, in_elem_count, in_elem_size);

        BufferPtr transformInput = BufferPtr::user_input(inBrickIdx);

        std::vector<size_t> antecedents;

        // on each brick, transform the contiguous dims
        for(auto dimIdx : contiguousDims)
        {
            auto transformItem = C2COneDimension(*this,
                                                 dimIdx,
                                                 inBrick.device,
                                                 inBrick.length(),
                                                 inBrick.stride,
                                                 transformInput,
                                                 BufferPtr::temp(temp.data()),
                                                 antecedents);

            transformInput = BufferPtr::temp(temp.data());
            antecedents    = {transformItem};
        }

        // need to transpose data so remaining dim is contiguous

        // for each output brick, work out the intersection of
        // this brick and transpose it if there's overlap
        for(size_t outBrickIdx = 0; outBrickIdx < desc.outFields.front().bricks.size();
            ++outBrickIdx)
        {
            const auto& outBrick = desc.outFields.front().bricks[outBrickIdx];

            auto intersection = inBrick.intersect(outBrick);
            if(intersection.empty())
                continue;
            intersection.stride = intersection.contiguous_strides();

            // pack data for communication
            TempBufferLease pack(
                tempBuffers, inBrick.device, intersection.count_elems(), in_elem_size);
            TempBufferLease recv(
                tempBuffers, outBrick.device, intersection.count_elems(), in_elem_size);
            auto packIdx
                = AddMultiPlanItem(transpose_brick(inBrick.device,
                                                   intersection.length(),
                                                   precision,
                                                   desc.inArrayType,
                                                   BufferPtr::temp(temp.data()),
                                                   intersection.offset_in_field(inBrick.stride)
                                                       - inBrick.offset_in_field(inBrick.stride),
                                                   inBrick.stride,
                                                   BufferPtr::temp(pack.data()),
                                                   0,
                                                   intersection.stride,
                                                   "pack brick for remaining dimension"),
                                   antecedents);
            antecedents = {packIdx};

            // send packed data
            auto sendOp          = std::make_unique<CommPointToPoint>();
            sendOp->precision    = precision;
            sendOp->arrayType    = desc.inArrayType;
            sendOp->numElems     = intersection.count_elems();
            sendOp->srcDeviceID  = inBrick.device;
            sendOp->srcPtr       = BufferPtr::temp(pack.data());
            sendOp->destDeviceID = outBrick.device;
            sendOp->destPtr      = BufferPtr::temp(recv.data());

            auto send   = AddMultiPlanItem(std::move(sendOp), antecedents);
            antecedents = {send};

            // unpack data on destination to user out
            auto unpackIdx
                = AddMultiPlanItem(transpose_brick(outBrick.device,
                                                   intersection.length(),
                                                   precision,
                                                   desc.inArrayType,
                                                   BufferPtr::temp(recv.data()),
                                                   0,
                                                   intersection.stride,
                                                   BufferPtr::user_output(outBrickIdx),
                                                   intersection.offset_in_field(outBrick.stride)
                                                       - outBrick.offset_in_field(outBrick.stride),
                                                   outBrick.stride,
                                                   "unpack brick for remaining dimension"),
                                   antecedents);
            unpackIndexes.push_back(unpackIdx);
        }
    }

    for(size_t outBrickIdx = 0; outBrickIdx < desc.outFields.front().bricks.size(); ++outBrickIdx)
    {
        const auto& outBrick = desc.outFields.front().bricks[outBrickIdx];

        // data is now transposed to output bricks.  fft
        // remaining dimension on all bricks
        C2COneDimension(*this,
                        nonContiguousDim,
                        outBrick.device,
                        outBrick.length(),
                        outBrick.stride,
                        BufferPtr::user_output(outBrickIdx),
                        BufferPtr::user_output(outBrickIdx),
                        unpackIndexes);
    }

    return true;
}

void rocfft_plan_t::ValidateFields() const
{
    auto validateField = [](const char*                type,
                            const std::vector<size_t>& lengths,
                            size_t                     batch,
                            const rocfft_field_t&      field) {
        // construct a brick that covers the whole field
        rocfft_brick_t whole_field;
        whole_field.lower.resize(lengths.size() + 1);
        whole_field.upper = lengths;
        whole_field.upper.push_back(batch);

        // ensure no bricks in the field overlap - check each brick
        // against each other brick
        for(auto brickI = field.bricks.begin(); brickI != field.bricks.end(); ++brickI)
        {
            for(auto brickJ = brickI + 1; brickJ != field.bricks.end(); ++brickJ)
            {
                if(!brickI->intersect(*brickJ).empty())
                {
                    throw std::runtime_error(std::string(type) + " brick " + brickI->str()
                                             + " overlaps with brick " + brickJ->str());
                }
            }

            // ensure each brick is within the field
            if(!brickI->equal_coords(brickI->intersect(whole_field)))
                throw std::runtime_error(std::string(type) + " brick " + brickI->str()
                                         + " is not within field");
        }

        // check that the bricks cover the whole index space
        const size_t whole_field_elems = whole_field.count_elems();

        // add up total number of elements in all bricks,
        // should be the same since we know there are no
        // overlaps
        size_t total_brick_elems = 0;
        for(const auto& b : field.bricks)
            total_brick_elems += b.count_elems();
        if(whole_field_elems != total_brick_elems)
            throw std::runtime_error(std::string(type) + " field has "
                                     + std::to_string(whole_field_elems) + " elems but bricks have "
                                     + std::to_string(total_brick_elems) + " elems");
    };

    if(!desc.inFields.empty())
        validateField("input", lengths, batch, desc.inFields.front());
    if(!desc.outFields.empty())
        validateField("output", outputLengths, batch, desc.outFields.front());
}

rocfft_status rocfft_plan_create_internal(rocfft_plan                   plan,
                                          const rocfft_result_placement placement,
                                          const rocfft_transform_type   transform_type,
                                          const rocfft_precision        precision,
                                          const size_t                  dimensions,
                                          const size_t*                 lengths,
                                          const size_t                  number_of_transforms,
                                          const rocfft_plan_description description)
{
    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;

    rocfft_plan p = plan;
    p->rank       = dimensions;
    std::copy(lengths, lengths + dimensions, std::back_inserter(p->lengths));
    p->batch         = number_of_transforms;
    p->placement     = placement;
    p->precision     = precision;
    p->transformType = transform_type;

    p->outputLengths = p->lengths;
    if(transform_type == rocfft_transform_type_real_forward
       || transform_type == rocfft_transform_type_real_inverse)
    {
        p->outputLengths.front() = p->outputLengths.front() / 2 + 1;
    }
    if(transform_type == rocfft_transform_type_real_inverse)
        std::swap(p->outputLengths, p->lengths);

    if(description != nullptr)
    {
        p->desc = *description;
    }
    p->desc.init_defaults(p->transformType, p->placement, p->lengths, p->outputLengths);

    // sort the parameters to be row major, in case they're not
    plan->sort();

    // Check plan validity
    switch(transform_type)
    {
    case rocfft_transform_type_complex_forward:
    case rocfft_transform_type_complex_inverse:
        // We need complex input data
        if(!((p->desc.inArrayType == rocfft_array_type_complex_interleaved)
             || (p->desc.inArrayType == rocfft_array_type_complex_planar)))
            return rocfft_status_invalid_array_type;
        // We need complex output data
        if(!((p->desc.outArrayType == rocfft_array_type_complex_interleaved)
             || (p->desc.outArrayType == rocfft_array_type_complex_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform requires that the input and output
        // format be identical
        if(placement == rocfft_placement_inplace)
        {
            if(p->desc.inArrayType != p->desc.outArrayType)
                return rocfft_status_invalid_array_type;
        }
        break;
    case rocfft_transform_type_real_forward:
        // Input must be real
        if(p->desc.inArrayType != rocfft_array_type_real)
            return rocfft_status_invalid_array_type;
        // Output must be Hermitian
        if(!((p->desc.outArrayType == rocfft_array_type_hermitian_interleaved)
             || (p->desc.outArrayType == rocfft_array_type_hermitian_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform must output to interleaved format
        if((placement == rocfft_placement_inplace)
           && (p->desc.outArrayType != rocfft_array_type_hermitian_interleaved))
            return rocfft_status_invalid_array_type;
        break;
    case rocfft_transform_type_real_inverse:
        // Output must be real
        if(p->desc.outArrayType != rocfft_array_type_real)
            return rocfft_status_invalid_array_type;
        // Input must be Hermitian
        if(!((p->desc.inArrayType == rocfft_array_type_hermitian_interleaved)
             || (p->desc.inArrayType == rocfft_array_type_hermitian_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform must have interleaved input
        if((placement == rocfft_placement_inplace)
           && (p->desc.inArrayType != rocfft_array_type_hermitian_interleaved))
            return rocfft_status_invalid_array_type;
        break;
    }

    log_bench(rocfft_bench_command(p));

    // construct the plan
    try
    {
        plan->ValidateFields();

        // build an optimized multi-device plan, if possible
        if(!plan->BuildOptMultiDevicePlan())
        {
            // if optimized multi-device was not possible (either because
            // multi-device was not requested, or we can't optimize for
            // that case), fall back to single-device plan

            NodeMetaData rootPlanData(nullptr);

            rootPlanData.dimension = plan->rank;
            rootPlanData.batch     = plan->batch;
            for(size_t i = 0; i < plan->rank; i++)
            {
                rootPlanData.length.push_back(plan->lengths[i]);
                rootPlanData.outputLength.push_back(plan->outputLengths[i]);

                rootPlanData.inStride.push_back(plan->desc.inStrides[i]);
                rootPlanData.outStride.push_back(plan->desc.outStrides[i]);
            }
            rootPlanData.iDist = plan->desc.inDist;
            rootPlanData.oDist = plan->desc.outDist;

            rootPlanData.placement = plan->placement;

            // If in+out fields are specified, currently that means we're
            // gathering the data to one device and doing FFT there.  So
            // the FFT becomes in-place.
            if(!plan->desc.inFields.empty() && !plan->desc.outFields.empty())
            {
                // c2c can be inplace so both input/output can be
                // contiguous without needing extra buffers
                if(plan->transformType == rocfft_transform_type_complex_forward
                   || plan->transformType == rocfft_transform_type_complex_inverse)
                {
                    rootPlanData.placement = rocfft_placement_inplace;
                }
                // real-complex would require non-contiguous data to be
                // inplace (which likely means more packing/unpacking and
                // extra temp buffer usage anyway), so make it
                // not-in-place instead
                else
                {
                    rootPlanData.placement = rocfft_placement_notinplace;
                }
            }
            // If we only have outfield, then there's no need to gather.  But we can't assume we can overwrite input, so the FFT will be outplace to a temp buf
            else if(plan->desc.inFields.empty() && !plan->desc.outFields.empty())
            {
                rootPlanData.placement = rocfft_placement_notinplace;
            }
            // If we only have infield, then we must gather.
            else if(!plan->desc.inFields.empty() && plan->desc.outFields.empty())
            {
                // If output is contiguous and this is c2c then we can
                // gather to the output buf and do an inplace FFT.
                if(plan->is_contiguous_output()
                   && (plan->transformType == rocfft_transform_type_complex_forward
                       || plan->transformType == rocfft_transform_type_complex_inverse))
                    rootPlanData.placement = rocfft_placement_inplace;
                // Otherwise we must gather to a temp buf and do outplace FFT to output
                else
                    rootPlanData.placement = rocfft_placement_notinplace;
            }

            rootPlanData.precision = plan->precision;
            if((plan->transformType == rocfft_transform_type_complex_forward)
               || (plan->transformType == rocfft_transform_type_real_forward))
                rootPlanData.direction = -1;
            else
                rootPlanData.direction = 1;

            rootPlanData.inArrayType  = plan->desc.inArrayType;
            rootPlanData.outArrayType = plan->desc.outArrayType;
            rootPlanData.rootIsC2C    = (rootPlanData.inArrayType != rocfft_array_type_real)
                                     && (rootPlanData.outArrayType != rocfft_array_type_real);

            set_bluestein_strides(plan, rootPlanData);

            rootPlanData.deviceProp = get_curr_device_prop();

            int deviceID;
            if(hipGetDevice(&deviceID) != hipSuccess)
                throw std::runtime_error("hipGetDevice failed");
            auto singleDevicePlan = BuildSingleDevicePlan(
                rootPlanData, deviceID, plan->transformType, p->desc.loadOps, p->desc.storeOps);

            p->GatherScatterSingleDevicePlan(std::move(singleDevicePlan));
        }
        return rocfft_status_success;
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }
}

rocfft_status rocfft_plan_allocate(rocfft_plan* plan)
{
    *plan = new rocfft_plan_t;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                 const rocfft_result_placement placement,
                                 const rocfft_transform_type   transform_type,
                                 const rocfft_precision        precision,
                                 const size_t                  dimensions,
                                 const size_t*                 lengths,
                                 const size_t                  number_of_transforms,
                                 const rocfft_plan_description description)
{
    rocfft_plan_allocate(plan);

    size_t log_len[3] = {1, 1, 1};
    if(dimensions > 0)
        log_len[0] = lengths[0];
    if(dimensions > 1)
        log_len[1] = lengths[1];
    if(dimensions > 2)
        log_len[2] = lengths[2];

    log_trace(__func__,
              "plan",
              *plan,
              "placement",
              placement,
              "transform_type",
              transform_type,
              "precision",
              precision,
              "dimensions",
              dimensions,
              "lengths",
              std::make_pair(lengths, dimensions),
              "number_of_transforms",
              number_of_transforms,
              "description",
              description);

    return rocfft_plan_create_internal(*plan,
                                       placement,
                                       transform_type,
                                       precision,
                                       dimensions,
                                       lengths,
                                       number_of_transforms,
                                       description);
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
{
    delete plan;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
{
    if(!plan)
        return rocfft_status_failure;

    *size_in_bytes = plan->WorkBufBytes();
    log_trace(__func__, "plan", plan, "size_in_bytes ptr", size_in_bytes, "val", *size_in_bytes);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    rocfft_cout << std::endl;
    rocfft_cout << "precision: " << precision_name(plan->precision) << std::endl;

    rocfft_cout << "transform type: ";
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
        rocfft_cout << "complex forward";
        break;
    case rocfft_transform_type_complex_inverse:
        rocfft_cout << "complex inverse";
        break;
    case rocfft_transform_type_real_forward:
        rocfft_cout << "real forward";
        break;
    case rocfft_transform_type_real_inverse:
        rocfft_cout << "real inverse";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "result placement: ";
    switch(plan->placement)
    {
    case rocfft_placement_inplace:
        rocfft_cout << "in-place";
        break;
    case rocfft_placement_notinplace:
        rocfft_cout << "not in-place";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input array type: ";
    switch(plan->desc.inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "complex planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "output array type: ";
    switch(plan->desc.outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "comple planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "dimensions: " << plan->rank << std::endl;

    rocfft_cout << "lengths: " << plan->lengths[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->lengths[i];
    rocfft_cout << std::endl;
    rocfft_cout << "batch size: " << plan->batch << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input offset: " << plan->desc.inOffset[0];
    if((plan->desc.inArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.inOffset[1];
    rocfft_cout << std::endl;

    rocfft_cout << "output offset: " << plan->desc.outOffset[0];
    if((plan->desc.outArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.outOffset[1];
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input strides: " << plan->desc.inStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.inStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "output strides: " << plan->desc.outStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.outStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "input distance: " << plan->desc.inDist << std::endl;
    rocfft_cout << "output distance: " << plan->desc.outDist << std::endl;
    rocfft_cout << std::endl;

    plan->desc.loadOps.print(rocfft_cout, {});
    plan->desc.storeOps.print(rocfft_cout, {});
    rocfft_cout << std::endl;

    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
{
    log_trace(__func__, "buf", static_cast<void*>(buf), "len", len);
    static constexpr char v[] = ROCFFT_VERSION_STRING;
    if(!buf)
        return rocfft_status_failure;
    if(len < sizeof(v))
        return rocfft_status_invalid_arg_value;
    memcpy(buf, v, sizeof(v));
    return rocfft_status_success;
}

// Compute the large twd decomposition base
// 2-Steps:
//  e.g., ( CeilPo2(10000)+ 1 ) / 2 , returns 7 : (2^7)*(2^7) = 16384 >= 10000
// 3-Steps:
//  e.g., ( CeilPo2(10000)+ 2 ) / 3 , returns 5 : (2^5)*(2^5)*(2^5) = 32768 >= 10000
void get_large_twd_base_steps(size_t large1DLen, bool use3steps, size_t& base, size_t& steps)
{
    // use3steps, then 16^3 ~ 64^3, basically enough for 262144
    // else, base is 8 (2^8 = 256), could be 2-steps 256^2 = 65536, if exceed, then is 256^3, and so on..
    base = use3steps ? std::min((size_t)6, std::max((size_t)4, (CeilPo2(large1DLen) + 2) / 3)) : 8;

    // but we still want to know the exact steps we will loop
    steps                  = 0;
    size_t lenLargeTwdBase = pow(2, base);
    while(pow(lenLargeTwdBase, steps) < large1DLen)
        steps++;

    if(base == 8 && steps > 3)
        throw std::runtime_error(
            "large-twd-base 8 could be 2,3 steps, but not supported for 4-steps yet");
    if(base < 8 && steps != 3)
        throw std::runtime_error("large-twd-base for 4,5,6 must be 3-steps");
}

bool BufferIsUnitStride(ExecPlan& execPlan, OperatingBuffer buf)
{
    // temp buffers are unit stride
    if(buf != OB_USER_IN && buf != OB_USER_OUT)
        return true;

    if(execPlan.isUnitStride.find(buf) != execPlan.isUnitStride.end())
        return execPlan.isUnitStride.at(buf);

    auto stride = (buf == OB_USER_IN) ? execPlan.rootPlan->inStride : execPlan.rootPlan->outStride;
    auto length = (buf == OB_USER_IN) ? execPlan.iLength : execPlan.oLength;
    auto dist   = (buf == OB_USER_IN) ? execPlan.rootPlan->iDist : execPlan.rootPlan->oDist;
    size_t curStride = 1;
    do
    {
        if(stride.front() != curStride)
            return false;
        curStride *= length.front();
        stride.erase(stride.begin());
        length.erase(length.begin());
    } while(!stride.empty());

    // NB: users may input incorrect i/o-dist value for inplace transform
    //     however, when the batch-size is 1, we can simply make it permissive
    //     since the dist is not used in single batch. But note that we still need
    //     to pass the above do-while to ensure all the previous strides are valid.
    bool result = (execPlan.rootPlan->batch == 1) || (curStride == dist);

    execPlan.isUnitStride[buf] = result;
    return result;
}

void TreeNode::CopyNodeData(const TreeNode& srcNode)
{
    dimension = srcNode.dimension;
    batch     = srcNode.batch;
    length    = srcNode.length;
    if(srcNode.outputLength != srcNode.length)
        outputLength = srcNode.outputLength;
    inStride        = srcNode.inStride;
    inStrideBlue    = srcNode.inStrideBlue;
    outStride       = srcNode.outStride;
    outStrideBlue   = srcNode.outStrideBlue;
    iDist           = srcNode.iDist;
    iDistBlue       = srcNode.iDistBlue;
    oDist           = srcNode.oDist;
    oDistBlue       = srcNode.oDistBlue;
    iOffset         = srcNode.iOffset;
    oOffset         = srcNode.oOffset;
    placement       = srcNode.placement;
    precision       = srcNode.precision;
    direction       = srcNode.direction;
    inArrayType     = srcNode.inArrayType;
    outArrayType    = srcNode.outArrayType;
    allowInplace    = srcNode.allowInplace;
    allowOutofplace = srcNode.allowOutofplace;
    deviceProp      = srcNode.deviceProp;

    // conditional
    large1D        = srcNode.large1D;
    largeTwd3Steps = srcNode.largeTwd3Steps;
    largeTwdBase   = srcNode.largeTwdBase;
    lengthBlue     = srcNode.lengthBlue;
    lengthBlueN    = srcNode.lengthBlueN;
    typeBlue       = srcNode.typeBlue;
    fuseBlue       = srcNode.fuseBlue;

    //
    obIn  = srcNode.obIn;
    obOut = srcNode.obOut;

    // NB:
    //   we don't copy these since it's possible we're copying
    //   a node to another one that is different scheme/derived class
    //   (for example, when doing fusion).
    //   The src ebtype could be incorrect in the new node
    //   same as lds_padding, lds_padding is initialized for each derived class
    //   so we don't copy this value, the target node already sets its value
    // ebtype      = srcNode.ebtype;
    // lds_padding = srcNode.lds_padding;
}

void TreeNode::CopyNodeData(const NodeMetaData& data)
{
    dimension = data.dimension;
    batch     = data.batch;
    length    = data.length;
    if(data.outputLength != data.length)
        outputLength = data.outputLength;
    inStride      = data.inStride;
    inStrideBlue  = data.inStrideBlue;
    outStride     = data.outStride;
    outStrideBlue = data.outStrideBlue;
    iDist         = data.iDist;
    iDistBlue     = data.iDistBlue;
    oDist         = data.oDist;
    oDistBlue     = data.oDistBlue;
    iOffset       = data.iOffset;
    oOffset       = data.oOffset;
    placement     = data.placement;
    precision     = data.precision;
    direction     = data.direction;
    inArrayType   = data.inArrayType;
    outArrayType  = data.outArrayType;
    deviceProp    = data.deviceProp;
}

bool TreeNode::isPlacementAllowed(rocfft_result_placement test_placement) const
{
    return (test_placement == rocfft_placement_inplace) ? allowInplace : allowOutofplace;
}

bool TreeNode::isOutBufAllowed(OperatingBuffer oB) const
{
    return (oB & allowedOutBuf) != 0;
}

bool TreeNode::isOutArrayTypeAllowed(rocfft_array_type oArrayType) const
{
    return allowedOutArrayTypes.count(oArrayType) > 0;
}

bool TreeNode::isRootNode() const
{
    return parent == nullptr;
}

bool TreeNode::isLeafNode() const
{
    return nodeType == NT_LEAF;
}

// Tree node builders

// NB:
// Don't assign inArrayType and outArrayType when building any tree node.
// That should be done in buffer assignment stage or
// TraverseTreeAssignPlacementsLogicA().

void TreeNode::RecursiveBuildTree(SchemeTree* solution_scheme)
{
    // Some-Common-Work...
    // We must follow the placement of RootPlan, so needs to make it explicit
    if(isRootNode())
    {
        allowInplace    = (placement == rocfft_placement_inplace);
        allowOutofplace = !allowInplace;
    }

    SchemeTreeVec& child_scheme
        = (solution_scheme) ? solution_scheme->children : EmptySchemeTreeVec;

    // overriden by each derived class
    BuildTree_internal(child_scheme);
}

void TreeNode::SanityCheck(SchemeTree* solution_scheme, std::vector<FMKey>& kernel_keys)
{
    // no un-defined node is allowed in the tree
    if(nodeType == NT_UNDEFINED)
        throw std::runtime_error("NT_UNDEFINED node");

    // Check buffer: all operating buffers have been assigned
    if(obIn == OB_UNINIT)
        throw std::runtime_error("obIn un-init");
    if(obOut == OB_UNINIT)
        throw std::runtime_error("obOut un-init");
    if((obIn == obOut) && (placement != rocfft_placement_inplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement inplace");
    if((obIn != obOut) && (placement != rocfft_placement_notinplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement out-of-place");

    // Check length and stride and dimension:
    if(length.size() != inStride.size())
        throw std::runtime_error("length.size() mismatch inStride.size()");
    if(length.size() != outStride.size())
        throw std::runtime_error("length.size() mismatch outStride.size()");
    if(length.size() < dimension)
        throw std::runtime_error("not enough length[] for dimension");

    // make sure the tree has the same decomposition way as in solution map
    if(solution_scheme)
    {
        if(childNodes.size() != solution_scheme->children.size())
            throw std::runtime_error("scheme-decomposition error: plan-tree != scheme-tree");
        if(scheme != solution_scheme->curScheme)
            throw std::runtime_error("scheme-decomposition error: node-scheme != solution-scheme");
    }

    OperatingBuffer previousOut = obIn;
    for(size_t id = 0; id < childNodes.size(); ++id)
    {
        auto&       child = childNodes[id];
        SchemeTree* child_scheme
            = (solution_scheme) ? solution_scheme->children[id].get() : nullptr;

        // 1. Recursively check child
        child->SanityCheck(child_scheme, kernel_keys);

        // 2. Assert that the kernel chain is connected
        // Note: The Bluestein algorithm uses setup nodes that aren't
        // connected in the chain.

        if(child->IsBluesteinChirpSetup())
            continue;
        if(child->obIn != previousOut)
            throw std::runtime_error("Sanity Check failed: " + PrintScheme(child->scheme)
                                     + " input " + PrintOperatingBuffer(child->obIn)
                                     + " does not match previous output "
                                     + PrintOperatingBuffer(previousOut));
        previousOut = child->obOut;
    }
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_Z_XY()
{
    if(function_pool::has_SBRC_kernel(length[0], precision))
    {
        auto kernel = function_pool::get_kernel(
            FMKey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC, TILE_ALIGNED));
        size_t bwd = kernel.transforms_per_block;
        if((length[1] >= bwd) && (length[2] >= bwd) && (length[1] * length[2] % bwd == 0))
            return true;
    }

    return false;
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_XY_Z()
{
    if(function_pool::has_SBRC_kernel(length[0], precision))
    {
        if((length[0] == length[2]) // limit to original "cubic" case
           && (length[0] / 2 + 1 == length[1])
           && !IsPo2(length[0]) // Need more investigation for diagonal transpose
        )
            return true;
    }
    return false;
}

bool TreeNode::fuse_CS_KERNEL_STK_R2C_TRANSPOSE()
{
    if(function_pool::has_SBRC_kernel(length[0], precision)) // kernel available
    {
        if((length[0] * 2 == length[1]) // limit to original "cubic" case
           && (length.size() == 2 || length[1] == length[2]) // 2D or 3D
        )
            return true;
    }
    return false;
}

void TreeNode::ApplyFusion()
{
    // Do the final fusion after the buffer assign is completed
    for(auto& fuse : fuseShims)
    {
        // the flag was overwritten by execPlan (according to the arch for some specical cases)
        if(!fuse->IsSchemeFusable())
            continue;

        auto fused = fuse->FuseKernels();
        if(fused)
        {
            auto firstFusedNode = fuse->FirstFuseNode();
            this->RecursiveInsertNode(firstFusedNode, fused);

            // iterate from first to last to remove old nodes
            fuse->ForEachNode([=](TreeNode* node) { this->RecursiveRemoveNode(node); });
        }
    }

    for(auto& child : childNodes)
        child->ApplyFusion();
}

void TreeNode::RefreshTree()
{
    if(childNodes.empty())
        return;

    for(auto& child : childNodes)
        child->RefreshTree();

    // only modify nodes that work with user data, and skip Bluestein
    // nodes that only set up the chirp buffer
    auto firstIt = std::find_if_not(
        childNodes.begin(), childNodes.end(), [](const std::unique_ptr<TreeNode>& n) {
            return n->IsBluesteinChirpSetup();
        });
    // if these children are all setup nodes, there's nothing further to refresh
    if(firstIt == childNodes.end())
        return;

    auto first = firstIt->get();
    auto last  = childNodes.back().get();

    // Skip first node in multi-kernel fused Bluestein
    // since it is not connected to the buffer chain
    if(fuseBlue != BFT_FWD_CHIRP)
    {
        this->obIn      = first->obIn;
        this->obOut     = last->obOut;
        this->placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;

        // even-length real transform nodes need to have real
        // input/output even if their first/last child treats the
        // real data as complex
        const bool isRealEvenNode = scheme == CS_REAL_TRANSFORM_EVEN || scheme == CS_REAL_2D_EVEN
                                    || scheme == CS_REAL_3D_EVEN;
        if(isRealEvenNode && direction == -1)
            this->inArrayType = rocfft_array_type_real;
        else
            this->inArrayType = first->inArrayType;

        if(isRealEvenNode && direction == 1)
            this->outArrayType = rocfft_array_type_real;
        else
            this->outArrayType = last->outArrayType;
    }
}

void TreeNode::AssignParams()
{
    if((length.size() != inStride.size()) || (length.size() != outStride.size()))
        throw std::runtime_error("length size mismatches stride size");

    for(auto& child : childNodes)
    {
        child->inStride.clear();
        child->inStrideBlue.clear();
        child->outStride.clear();
        child->outStrideBlue.clear();
    }

    AssignParams_internal();
}

///////////////////////////////////////////////////////////////////////////////
/// Collect leaf node
void TreeNode::CollectLeaves(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    // re-collect after kernel fusion, so clear the previous collected elements
    if(isRootNode())
    {
        seq.clear();
        fuseSeq.clear();
    }

    if(nodeType == NT_LEAF)
    {
        seq.push_back(this);
    }
    else
    {
        for(auto& child : childNodes)
            child->CollectLeaves(seq, fuseSeq);

        for(auto& fuse : fuseShims)
            fuseSeq.push_back(fuse.get());
    }
}

// Important: Make sure the order of the fuse-shim is consistent with the execSeq
// This is essential for BackTracking in BufferAssignment
void OrderFuseShims(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    std::vector<FuseShim*> reordered;
    for(auto node : seq)
    {
        for(size_t fuseID = 0; fuseID < fuseSeq.size(); ++fuseID)
        {
            if(node == fuseSeq[fuseID]->FirstFuseNode())
            {
                reordered.emplace_back(fuseSeq[fuseID]);
                break;
            }
        }
    }

    if(reordered.size() != fuseSeq.size())
        throw std::runtime_error("reorder fuse shim list error");

    fuseSeq.swap(reordered);
}

void CheckFuseShimForArch(ExecPlan& execPlan)
{
    // for gfx906...
    if(is_device_gcn_arch(execPlan.deviceProp, "gfx906"))
    {
        auto& fusions = execPlan.fuseShims;
        for(auto& fusion : fusions)
        {
            if(fusion->fuseType == FT_STOCKHAM_WITH_TRANS
               && fusion->FirstFuseNode()->length[0] == 168)
            {
                fusion->OverwriteFusableFlag(false);

                // remove it from the execPlan list
                fusions.erase(std::remove(fusions.begin(), fusions.end(), fusion), fusions.end());
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Calculate work memory requirements,
/// note this should be done after buffer assignment and deciding oDist
void TreeNode::DetermineBufferMemory(size_t& tmpBufSize,
                                     size_t& cmplxForRealSize,
                                     size_t& blueSize,
                                     size_t& chirpSize)
{
    if(nodeType == NT_LEAF)
    {
        auto outputPtrDiff
            = compute_ptrdiff(UseOutputLengthForPadding() ? GetOutputLength() : length,
                              (typeBlue == BT_MULTI_KERNEL_FUSED) ? outStrideBlue : outStride,
                              batch,
                              (typeBlue == BT_MULTI_KERNEL_FUSED) ? oDistBlue : oDist);

        if(scheme == CS_KERNEL_CHIRP)
            chirpSize = std::max(lengthBlue, chirpSize);

        if(obOut == OB_TEMP_BLUESTEIN)
            blueSize = std::max(typeBlue == BT_MULTI_KERNEL_FUSED ? outputPtrDiff + lengthBlue
                                                                  : outputPtrDiff,
                                blueSize);

        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
            cmplxForRealSize = std::max(outputPtrDiff, cmplxForRealSize);

        if(obOut == OB_TEMP)
            tmpBufSize = std::max(outputPtrDiff, tmpBufSize);
    }

    for(auto& child : childNodes)
        child->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
}

void TreeNode::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << "\n" << indentStr << "scheme: " << PrintScheme(scheme);
    os << "\n" << indentStr;
    os << "dimension: " << dimension;
    os << "\n" << indentStr;
    os << "batch: " << batch;
    os << "\n" << indentStr;
    os << "length: ";
    for(size_t i = 0; i < length.size(); i++)
    {
        os << length[i] << " ";
    }
    if(!outputLength.empty() && outputLength != length)
    {
        os << "\n" << indentStr;
        os << "outputLength: ";
        for(size_t i = 0; i < outputLength.size(); i++)
        {
            os << outputLength[i] << " ";
        }
    }

    os << "\n" << indentStr << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << " ";

    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr << "iStridesBlue: ";
        for(size_t i = 0; i < inStrideBlue.size(); i++)
            os << inStrideBlue[i] << " ";
    }

    os << "\n" << indentStr << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << " ";

    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr << "oStridesBlue: ";
        for(size_t i = 0; i < outStrideBlue.size(); i++)
            os << outStrideBlue[i] << " ";
    }

    if(iOffset)
    {
        os << "\n" << indentStr;
        os << "iOffset: " << iOffset;
    }
    if(oOffset)
    {
        os << "\n" << indentStr;
        os << "oOffset: " << oOffset;
    }

    os << "\n" << indentStr;
    os << "iDist: " << iDist;
    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr;
        os << "iDistBlue: " << iDistBlue;
    }
    os << "\n" << indentStr;
    os << "oDist: " << oDist;
    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr;
        os << "oDistBlue: " << oDistBlue;
    }

    os << "\n" << indentStr;
    os << "direction: " << direction;

    os << "\n" << indentStr;
    os << "placement: " << PrintPlacement(placement);

    os << "\n" << indentStr;
    os << precision_name(precision) << "-precision";

    os << std::endl << indentStr;
    os << "array type: ";
    os << PrintArrayType(inArrayType);
    os << " -> ";
    os << PrintArrayType(outArrayType);

    if(large1D)
    {
        os << "\n" << indentStr << "large1D: " << large1D;
        os << "\n" << indentStr << "largeTwdBase: " << largeTwdBase;
        os << "\n" << indentStr << "largeTwdSteps: " << ltwdSteps;
    }
    if(twiddles)
    {
        os << "\n"
           << indentStr << "twiddle table length: " << twiddles_size / complex_type_size(precision);
    }
    if(twiddles_large)
    {
        os << "\n"
           << indentStr
           << "large twiddle table length: " << twiddles_large_size / complex_type_size(precision);
    }
    if(lengthBlue)
        os << "\n" << indentStr << "lengthBlue: " << lengthBlue;
    os << "\n";
    switch(ebtype)
    {
    case EmbeddedType::NONE:
        break;
    case EmbeddedType::C2Real_PRE:
        os << indentStr << "EmbeddedType: C2Real_PRE\n";
        break;
    case EmbeddedType::Real2C_POST:
        os << indentStr << "EmbeddedType: Real2C_POST\n";
        break;
    }

    os << indentStr << "SBRC_Trans_Type: " << PrintSBRCTransposeType(sbrcTranstype);
    os << "\n";

    switch(intrinsicMode)
    {
    case IntrinsicAccessType::DISABLE_BOTH:
        break;
    case IntrinsicAccessType::ENABLE_LOAD_ONLY:
        os << indentStr << "Intrinsic Mode: LOAD_ONLY\n";
        break;
    case IntrinsicAccessType::ENABLE_BOTH:
        os << indentStr << "Intrinsic Mode: LOAD_AND_STORE\n";
        break;
    }

    os << indentStr << "Direct_to_from_Reg: " << PrintDirectToFromRegMode(dir2regMode);
    os << "\n";
    loadOps.print(os, indentStr);
    storeOps.print(os, indentStr);

    os << indentStr << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n";
    os << indentStr << PrintOperatingBufferCode(obIn) << " -> " << PrintOperatingBufferCode(obOut)
       << "\n";
    for(const auto& c : comments)
    {
        os << indentStr << "comment: " << c << "\n";
    }

    if(childNodes.size())
    {
        for(auto& children_p : childNodes)
        {
            children_p->Print(os, indent + 1);
        }
    }
    os << std::flush;
}

void TreeNode::RecursiveFindChildNodes(const ComputeScheme&    findScheme,
                                       std::vector<TreeNode*>& nodes)
{
    if(scheme == findScheme)
        nodes.emplace_back(this);

    for(auto& child : childNodes)
        child->RecursiveFindChildNodes(findScheme, nodes);
}

void TreeNode::RecursiveCopyNodeData(const TreeNode& srcNode)
{
    CopyNodeData(srcNode);

    if(childNodes.size() != srcNode.childNodes.size())
        throw std::runtime_error("Invalid copy of source tree data");

    std::size_t i = 0;
    for(auto& child : childNodes)
    {
        child->CopyNodeData(*srcNode.childNodes[i]);
        ++i;
    }
}

void TreeNode::RecursiveRemoveNode(TreeNode* node)
{
    for(auto& child : childNodes)
        child->RecursiveRemoveNode(node);
    childNodes.erase(std::remove_if(childNodes.begin(),
                                    childNodes.end(),
                                    [node](const std::unique_ptr<TreeNode>& child) {
                                        return child.get() == node;
                                    }),
                     childNodes.end());
}

void TreeNode::RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto found = std::find_if(
        childNodes.begin(), childNodes.end(), [pos](const std::unique_ptr<TreeNode>& child) {
            return child.get() == pos;
        });
    if(found != childNodes.end())
    {
        childNodes.insert(found, std::move(newNode));
    }
    else
    {
        for(auto& child : childNodes)
            child->RecursiveInsertNode(pos, newNode);
    }
}

TreeNode* TreeNode::GetPlanRoot()
{
    if(isRootNode())
        return this;

    return parent->GetPlanRoot();
}

TreeNode* TreeNode::GetFirstLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.front()->GetFirstLeaf();
}

TreeNode* TreeNode::GetLastLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.back()->GetLastLeaf();
}

bool TreeNode::IsRootPlanC2CTransform()
{
    auto root = GetPlanRoot();
    return (root->inArrayType != rocfft_array_type_real)
           && (root->outArrayType != rocfft_array_type_real);
}

// remove a leaf node from the plan completely - plan optimization
// can remove unnecessary nodes to skip unnecessary work.
void RemoveNode(ExecPlan& execPlan, TreeNode* node)
{
    auto& execSeq = execPlan.execSeq;
    // remove it from the non-owning leaf nodes
    execSeq.erase(std::remove(execSeq.begin(), execSeq.end(), node), execSeq.end());

    // remove it from the tree structure
    execPlan.rootPlan->RecursiveRemoveNode(node);
}

// insert a leaf node to the plan, bot execSeq and tree - plan optimization
void InsertNode(ExecPlan& execPlan, TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto& execSeq = execPlan.execSeq;
    // insert it to execSeq, before pos
    execSeq.insert(std::find(execSeq.begin(), execSeq.end(), pos), newNode.get());

    // insert it before pos in the tree structure
    execPlan.rootPlan->RecursiveInsertNode(pos, newNode);
}

std::pair<TreeNode*, TreeNode*> ExecPlan::get_load_store_nodes() const
{
    const auto& seq = execSeq;

    // look forward for the first node that reads from input
    auto load_it = std::find_if(
        seq.begin(), seq.end(), [&](const TreeNode* n) { return n->obIn == rootPlan->obIn; });
    TreeNode* load = load_it == seq.end() ? nullptr : *load_it;

    // look backward for the last node that writes to output
    auto store_it = std::find_if(
        seq.rbegin(), seq.rend(), [&](const TreeNode* n) { return n->obOut == rootPlan->obOut; });
    TreeNode* store = store_it == seq.rend() ? nullptr : *store_it;

    assert(load && store);
    return std::make_pair(load, store);
}

void RuntimeCompilePlan(ExecPlan& execPlan)
{
    std::string kernel_name;
    bool        is_tuning = TuningBenchmarker::GetSingleton().IsProcessingTuning();

    for(size_t i = 0; i < execPlan.execSeq.size(); ++i)
    {
        auto& node = execPlan.execSeq[i];

        node->compiledKernel
            = RTCKernel::runtime_compile(*node, execPlan.deviceProp.gcnArchName, kernel_name);

        // Log kernel name when tuning
        if(is_tuning)
        {
            TuningBenchmarker::GetSingleton().GetPacket()->kernel_names[i] = kernel_name;
            if(LOG_TUNING_ENABLED())
                (*LogSingleton::GetInstance().GetTuningOS())
                    << "kernel: " << kernel_name << std::endl;
        }
    }

    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = execPlan.get_load_store_nodes();

    // callbacks are only possible on plans that don't use planar format for input or output
    bool need_callbacks = !array_type_is_planar(load_node->inArrayType)
                          && !array_type_is_planar(store_node->outArrayType);

    // don't spend time compiling callback
    if(need_callbacks && !is_tuning)
    {
        load_node->compiledKernelWithCallbacks = RTCKernel::runtime_compile(
            *load_node, execPlan.deviceProp.gcnArchName, kernel_name, true);

        if(store_node != load_node)
        {
            store_node->compiledKernelWithCallbacks = RTCKernel::runtime_compile(
                *store_node, execPlan.deviceProp.gcnArchName, kernel_name, true);
        }
    }

    // All of the compilations are started in parallel (via futures),
    // so resolve the futures now.  That ensures that the plan is
    // ready to run as soon as the caller gets the plan back.
    for(auto& node : execPlan.execSeq)
    {
        if(node->compiledKernel.valid())
            node->compiledKernel.get();
        if(node->compiledKernelWithCallbacks.valid())
            node->compiledKernelWithCallbacks.get();
    }
}

// Input a node, get the representative prob-token as the key of solution-map
void GetNodeToken(const TreeNode& probNode, std::string& min_token, std::string& full_token)
{
    // min_token: consider only length, precision, placement, complex/real,
    //             and direction for real-trans (R2C/C2R)
    // full_token: consider batch, dist, stride, offset, direction for complex
    // When searching solution, looking for full-match first, and then min-match

    // if this is a leaf-node TRANSPOSE, call_back or others with external-kernel = false
    // currently we don't tune it, but still need to put an entry in the map. So we
    // set a pre-defined token
    if(probNode.isLeafNode() && probNode.GetKernelKey() == FMKey::EmptyFMKey())
    {
        min_token = full_token = solution_map::LEAFNODE_TOKEN_BUILTIN_KERNEL;
        return;
    }

    std::string token = ComputeSchemeIsAProblem(probNode.scheme)
                            ? ("")
                            : (PrintKernelSchemeAbbr(probNode.scheme) + "_");

    for(size_t i = 0; i < probNode.dimension; ++i)
        token += std::to_string(probNode.length[i]) + "_";

    std::string precision_str;
    if(probNode.precision == rocfft_precision_single)
        precision_str = "sp_";
    else if(probNode.precision == rocfft_precision_double)
        precision_str = "dp_";
    else if(probNode.precision == rocfft_precision_half)
        precision_str = "half_";
    else
        throw std::runtime_error("tree node has invalid precision");

    token += precision_str;
    token += (probNode.placement == rocfft_placement_inplace) ? "ip_" : "op_";

    bool is_real_trans = ((probNode.inArrayType == rocfft_array_type_real)
                          || (probNode.outArrayType == rocfft_array_type_real));
    bool is_fwd        = (probNode.direction == -1);

    if(is_real_trans)
    {
        token += "real_";
        token += (is_fwd) ? "fwd" : "bwd";
        min_token = token;
    }
    else
    {
        token += "complex";
        min_token = token;
        token += (is_fwd) ? "_fwd" : "_bwd";
    }

    token += "_batch_" + std::to_string(probNode.batch);

    token += "_istride";
    for(size_t i = 0; i < probNode.inStride.size(); ++i)
        token += "_" + std::to_string(probNode.inStride[i]);

    token += "_ostride";
    for(size_t i = 0; i < probNode.outStride.size(); ++i)
        token += "_" + std::to_string(probNode.outStride[i]);

    token += "_idist_" + std::to_string(probNode.iDist);
    token += "_odist_" + std::to_string(probNode.oDist);
    token += "_ioffset_" + std::to_string(probNode.iOffset);
    token += "_ooffset_" + std::to_string(probNode.oOffset);

    full_token = token;
}

// generate all possible keys from a root problem, try them all to find a solution.
void GenerateProbKeys(const TreeNode& probNode, std::vector<ProblemKey>& possibleKeys)
{
    possibleKeys.clear();

    std::string min_token;
    std::string full_token;
    std::string archName = get_arch_name(probNode.deviceProp);
    GetNodeToken(probNode, min_token, full_token);

    for(auto arch : {archName, std::string("any")})
    {
        for(auto prob_token : {full_token, min_token})
        {
            ProblemKey problemKey(arch, prob_token);
            possibleKeys.push_back(problemKey);
        }
    }
}

// recursively apply the solutions (breadth-first)
// return: A pointer of a sub-scheme-tree
// If solution is a kernel, append the kernel_key to the output vector
std::unique_ptr<SchemeTree>
    RecursivelyApplySol(const ProblemKey& problemKey, ExecPlan& execPlan, size_t sol_option)
{
    auto& sol_map_single = solution_map::get_solution_map();
    if(!sol_map_single.has_solution_node(problemKey, sol_option))
        return nullptr;

    std::string  arch     = problemKey.arch;
    SolutionNode sol_node = sol_map_single.get_solution_node(problemKey, sol_option);

    // it is a dummy solution.
    if(sol_node.using_scheme == CS_NONE)
    {
        if(LOG_TRACE_ENABLED())
            (*LogSingleton::GetInstance().GetTraceOS())
                << "found a dummy root-solution(" << arch << ", " << problemKey.probToken << ")"
                << std::endl;
        return nullptr;
    }

    std::unique_ptr<SchemeTree> curScheme
        = std::make_unique<SchemeTree>(SchemeTree(sol_node.using_scheme));

    if(sol_node.sol_node_type == SOL_INTERNAL_NODE)
    {
        if(sol_node.solution_childnodes.empty())
            return nullptr;

        // we stick to the current arch same as the root's problemkey
        // e.g even we are in gfx908, but if the found root solution is in "any" map,
        // then we should keep looking-up the "any" map
        for(auto& child_node : sol_node.solution_childnodes)
        {
            ProblemKey probKey(arch, child_node.child_token);
            auto childScheme = RecursivelyApplySol(probKey, execPlan, child_node.child_option);
            if(!childScheme)
                return nullptr;

            curScheme->numKernels += childScheme->numKernels;
            curScheme->children.emplace_back(std::move(childScheme));
        }
    }
    // SOL_LEAF_NODE
    else if(sol_node.sol_node_type == SOL_LEAF_NODE)
    {
        // a leaf node should have exactly one child sol-node (SOL_KERNEL_ONLY or SOL_BUILTIN_KERNEL)
        if(sol_node.solution_childnodes.size() != 1)
            return nullptr;

        std::string& kernel_token   = sol_node.solution_childnodes[0].child_token;
        size_t       kernel_option  = sol_node.solution_childnodes[0].child_option;
        bool         tunable_kernel = (kernel_token != solution_map::KERNEL_TOKEN_BUILTIN_KERNEL);

        // When tuning, we're runing through each bench
        // so we use the elaborated token (_leafnode_id_phase_id)
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning() && tunable_kernel)
        {
            auto tuningPacket          = TuningBenchmarker::GetSingleton().GetPacket();
            int  curr_tuning_node_id   = tuningPacket->tuning_node_id;
            int  curr_tuning_phase     = tuningPacket->tuning_phase;
            int  curr_tuning_config_id = tuningPacket->current_ssn;

            // replacing the tuning target kernel_token to the candidate version
            size_t cur_leaf_node_id = execPlan.solution_kernels.size();
            kernel_token += "_leafnode_" + std::to_string(cur_leaf_node_id);

            if(cur_leaf_node_id == (size_t)curr_tuning_node_id)
            {
                // if this kernel is the one we're tuning, then we set the testing phase and config_id
                kernel_token += "_phase_" + std::to_string(curr_tuning_phase);
                kernel_option = curr_tuning_config_id;
            }
            else
            {
                // if the kernel is not the tuning target: we should fix the kernel to the current winner
                int curWinnerPhase = tuningPacket->winner_phases[cur_leaf_node_id];
                int curWinnerID    = tuningPacket->winner_ids[cur_leaf_node_id];

                kernel_token += "_phase_" + std::to_string(curWinnerPhase);
                kernel_option = curWinnerID;
            }
        }

        ProblemKey probKey_kernel(arch, kernel_token);
        if(!sol_map_single.has_solution_node(probKey_kernel, kernel_option))
            return nullptr;

        // get the kernel of this leaf node, be sure to pick the right kernel option
        SolutionNode& kernel_node = sol_map_single.get_solution_node(probKey_kernel, kernel_option);
        execPlan.solution_kernels.push_back(kernel_node.kernel_key);
        curScheme->numKernels = 1;

        // Keep the references, and after buffer-assignment and colapse-batch-dim,
        // we can save some info back to the kernel-configurations
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning())
        {
            execPlan.sol_kernel_configs.push_back(&(kernel_node.kernel_key.kernel_config));
        }

        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS())
                << "found the kernel solution(" << arch << ", " << kernel_token
                << ") with option: " << kernel_option << std::endl;
        }
    }
    // we shouldn't handle any SOL_KERNEL_ONLY directly
    else
    {
        throw std::runtime_error("Tree-Decomposition in solution map is invalid");
        return nullptr;
    }

    // if here, means we've found valid solutions of all sub-probs
    if(LOG_TRACE_ENABLED())
    {
        (*LogSingleton::GetInstance().GetTraceOS())
            << "found solution for problemKey(" << problemKey.arch << ", " << problemKey.probToken
            << ") with option: " << sol_option << std::endl;
    }
    if(LOG_TUNING_ENABLED())
    {
        (*LogSingleton::GetInstance().GetTuningOS())
            << "[SolToken]: " << problemKey.probToken << std::endl;
    }

    return curScheme;
}

std::unique_ptr<SchemeTree> ApplySolution(ExecPlan& execPlan)
{
    std::vector<ProblemKey>     possibleKeys;
    std::unique_ptr<SchemeTree> rootNodeScheme = nullptr;
    GenerateProbKeys(*(execPlan.rootPlan), possibleKeys);

    for(const auto& probKey : possibleKeys)
    {
        // found a valid solution-tree-decomposition
        rootNodeScheme = RecursivelyApplySol(probKey, execPlan, 0);
        if(rootNodeScheme)
            break;

        execPlan.solution_kernels = EmptyFMKeyVec;
    }

    return rootNodeScheme;
}

void ProcessNode(ExecPlan& execPlan)
{
    SchemeTree* rootScheme = (execPlan.rootScheme) ? execPlan.rootScheme.get() : nullptr;
    bool        noSolution = (rootScheme == nullptr);

    execPlan.rootPlan->RecursiveBuildTree(rootScheme);

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    // collect leaf-nodes to execSeq and fuseShims
    execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);

    if(noSolution)
    {
        CheckFuseShimForArch(execPlan);
        OrderFuseShims(execPlan.execSeq, execPlan.fuseShims);
    }

    // initialize root plan input/output location if not already done
    if(execPlan.rootPlan->obOut == OB_UNINIT)
        execPlan.rootPlan->obOut = OB_USER_OUT;
    if(execPlan.rootPlan->obIn == OB_UNINIT)
        execPlan.rootPlan->obIn
            = execPlan.rootPlan->placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;

    // guarantee min buffers but possible less fusions
    // execPlan.assignOptStrategy = rocfft_optimize_min_buffer;
    // starting from ABT
    execPlan.assignOptStrategy = rocfft_optimize_balance;
    // try to use all buffer to get most fusion
    //execPlan.assignOptStrategy = rocfft_optimize_max_fusion;
    AssignmentPolicy policy;
    policy.AssignBuffers(execPlan);

    if(TuningBenchmarker::GetSingleton().IsProcessingTuning() == false)
    {
        // Apply the fusion after buffer, strides are assigned
        execPlan.rootPlan->ApplyFusion();

        // collect the execSeq since we've fused some kernels
        execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);
    }

    // So we also need to update the whole tree including internal nodes
    // NB: The order matters: assign param -> fusion -> refresh internal node param
    execPlan.rootPlan->RefreshTree();

    // add padding if necessary
    policy.PadPlan(execPlan);

    // Collapse high dims on leaf nodes where possible
    execPlan.rootPlan->CollapseContiguousDims();

    // Check the buffer, param and tree integrity, Note we do this after fusion
    try
    {
        // rootScheme might be nullptr and solution_kernels might be empty (when no solution)
        // if has solution, will also check if it's valid
        execPlan.rootPlan->SanityCheck(rootScheme, execPlan.solution_kernels);
    }
    catch(const std::exception& e)
    {
        // When SanityCheck fails,
        // if solution_kernels is empty or rootScheme is nullptr,
        // means this is nothing to do with solution map. Throw to terminate
        if(execPlan.solution_kernels.empty() || rootScheme == nullptr)
            throw;
        else
        {
            // data from solution map are invalid, then we're not able to use them
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "input solution are invalid, try replacing kernels" << std::endl;
            execPlan.rootPlan->SanityCheck();
        }
    }

    // get workBufSize..
    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

    if(execPlan.rootPlan->loadOps.enabled())
    {
        // Load ops happen on first node that reads input
        auto load_node = std::find_if(
            execPlan.execSeq.begin(), execPlan.execSeq.end(), [&execPlan](TreeNode* node) {
                return node->obIn == execPlan.rootPlan->obIn;
            });
        (*load_node)->loadOps = execPlan.rootPlan->loadOps;
    }

    if(execPlan.rootPlan->storeOps.enabled())
    {
        // Store ops happen on last node of the plan that writes
        // output
        auto store_node = std::find_if(
            execPlan.execSeq.rbegin(), execPlan.execSeq.rend(), [&execPlan](TreeNode* node) {
                return node->obOut == execPlan.rootPlan->obOut;
            });
        (*store_node)->storeOps = execPlan.rootPlan->storeOps;
    }

    // compile kernels for applicable nodes
    RuntimeCompilePlan(execPlan);

    execPlan.workBufSize      = tmpBufSize + cmplxForRealSize + blueSize + chirpSize;
    execPlan.tmpWorkBufSize   = tmpBufSize;
    execPlan.copyWorkBufSize  = cmplxForRealSize;
    execPlan.blueWorkBufSize  = blueSize;
    execPlan.chirpWorkBufSize = chirpSize;
}

void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan, const int indent)
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << indentStr
       << "**********************************************************************"
          "*********"
       << std::endl;

    const size_t N = std::accumulate(execPlan.rootPlan->length.begin(),
                                     execPlan.rootPlan->length.end(),
                                     execPlan.rootPlan->batch,
                                     std::multiplies<size_t>());
    os << indentStr << "Work buffer size: " << execPlan.workBufSize << std::endl;
    os << indentStr << "Work buffer ratio: " << (double)execPlan.workBufSize / (double)N
       << std::endl;
    os << indentStr << "Assignment strategy: " << PrintOptimizeStrategy(execPlan.assignOptStrategy)
       << std::endl;

    execPlan.rootPlan->Print(os, indent);

    os << indentStr << "GridParams\n";
    for(const auto& gp : execPlan.gridParam)
    {
        os << indentStr << "  b[" << gp.b_x << "," << gp.b_y << "," << gp.b_z << "] wgs["
           << gp.wgs_x << "," << gp.wgs_y << "," << gp.wgs_z << "], dy_lds bytes " << gp.lds_bytes
           << "\n";
    }
    os << indentStr << "End GridParams\n";

    os << indentStr
       << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
