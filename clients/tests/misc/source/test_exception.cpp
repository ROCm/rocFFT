/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "test_exception.h"
#include <gtest/gtest.h>
#include <iostream>
#include <string>

void handle_exception(const std::exception& except)
{
    std::cout << "--- EXCEPTION CAUGHT ---" << std::endl;
    std::string error_message = except.what();
    std::cout << error_message << std::endl;
    FAIL();
}
