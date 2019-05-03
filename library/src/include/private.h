/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef __ROCFFT_PRIVATE_H__
#define __ROCFFT_PRIVATE_H__

#define DLL_PUBLIC __attribute__((visibility("default")))
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

DLL_PUBLIC rocfft_status rocfft_plan_create_internal(rocfft_plan             plan,
                                                     rocfft_result_placement placement,
                                                     rocfft_transform_type   transform_type,
                                                     rocfft_precision        precision,
                                                     size_t                  dimensions,
                                                     const size_t*           lengths,
                                                     size_t                  number_of_transforms,
                                                     const rocfft_plan_description description,
                                                     bool                          dry_run);

// plan allocation only
DLL_PUBLIC rocfft_status rocfft_plan_allocate(rocfft_plan* plan);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __ROCFFT_PRIVATE_H__
