/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
    return rocfft_status_success;
    // return rocfft_plan_description_set_scale_float(description, scale);
}

template <>
rocfft_status rocfft_set_scale_template<double>(const rocfft_plan_description description,
                                                const double                  scale)
{
    return rocfft_status_success;
    // return rocfft_plan_description_set_scale_double(description, scale);
}
