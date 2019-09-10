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

#include <gtest/gtest.h>
#include <stdexcept>
#include <unistd.h>

#include "rocfft_transform.h"

// template specialization of two templates in rocfft_transform.h
template <>
rocfft_status rocfft_plan_create_template<float>(rocfft_plan*                  plan,
                                                 rocfft_result_placement       placement,
                                                 rocfft_transform_type         transform_type,
                                                 size_t                        dimensions,
                                                 const size_t*                 lengths,
                                                 size_t                        number_of_transforms,
                                                 const rocfft_plan_description description)
{
    return rocfft_plan_create(plan,
                              placement,
                              transform_type,
                              rocfft_precision_single,
                              dimensions,
                              lengths,
                              number_of_transforms,
                              description);
}

template <>
rocfft_status rocfft_plan_create_template<double>(rocfft_plan*            plan,
                                                  rocfft_result_placement placement,
                                                  rocfft_transform_type   transform_type,
                                                  size_t                  dimensions,
                                                  const size_t*           lengths,
                                                  size_t                  number_of_transforms,
                                                  const rocfft_plan_description description)
{
    return rocfft_plan_create(plan,
                              placement,
                              transform_type,
                              rocfft_precision_double,
                              dimensions,
                              lengths,
                              number_of_transforms,
                              description);
}

template <>
rocfft_status rocfft_set_scale_template<float>(const rocfft_plan_description description,
                                               const float                   scale)
{

    // TODO: enable this when this is enabled in rocfft.h
    //return rocfft_plan_description_set_scale_float(description, scale);
    return rocfft_status_success;
}

template <>
rocfft_status rocfft_set_scale_template<double>(const rocfft_plan_description description,
                                                const double                  scale)
{
    // TODO: enable this when this is enabled in rocfft.h
    // return rocfft_plan_description_set_scale_double(description, scale);
    return rocfft_status_success;
}
