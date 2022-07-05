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

#include "rtc.h"
#include <iostream>
#include <iterator>

int main(int argc, const char* const* argv)
{
    try
    {
        // collect stdin as kernel source
        std::string kernel_src;
        std::noskipws(std::cin);
        std::copy(std::istream_iterator<char>(std::cin),
                  std::istream_iterator<char>(),
                  std::back_inserter(kernel_src));

        // compile and write code object to stdout
        auto code = RTCKernel::compile_inprocess(kernel_src);
        std::cout.write(code.data(), code.size());
        return 0;
    }
    catch(std::exception& e)
    {
        // write exception content to stdout - exit status will tell
        // the caller that stdout is an error message instead of
        // compiled code
        std::cout << e.what();
        return 1;
    }
}
