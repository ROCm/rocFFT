/*******************************************************************************
 * Copyright (C) 2018 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <unistd.h>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "hipfft.h"
#include "test_constants.h"
#include "rocfft_against_fftw.h"


TEST(hipfftTest, Create1dPlan)
{

    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t length = 1024;
    EXPECT_TRUE(hipfftPlan1d(&plan, length, HIPFFT_C2C, 1) == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}
