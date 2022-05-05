// Copyright (C) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef __EXAMPLEUTILS_H__
#define __EXAMPLEUTILS_H__

// Increment the index (column-major) for looping over arbitrary dimensional loops with
// dimensions length.
template <class T1, class T2>
bool increment_colmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = 0; idim < length.size(); ++idim)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            break;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist.
template <class Tdata, class Tint1, class Tint2>
void printbuffer(const std::vector<Tdata>& data,
                 const std::vector<Tint1>& length,
                 const std::vector<Tint2>& stride,
                 const size_t              nbatch,
                 const size_t              dist)
{
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<size_t> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const auto i = std::inner_product(index.begin(), index.end(), stride.begin(), b * dist);
            assert(i >= 0);
            assert(i < data.size());
            std::cout << data[i] << " ";
            for(size_t idx = 0; idx < index.size(); ++idx)
            {
                if(index[idx] == (length[idx] - 1))
                {
                    std::cout << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_colmajor(index, length));
        std::cout << std::endl;
    }
}

// Check that an multi-dimensional array of complex values with dimensions length
// and straide stride, with nbatch copies separated by dist is Hermitian-symmetric.
template <class Tfloat, class Tint1, class Tint2>
bool check_symmetry(const std::vector<std::complex<Tfloat>>& data,
                    const std::vector<Tint1>&                length,
                    const std::vector<Tint2>&                stride,
                    const size_t                             nbatch,
                    const size_t                             dist)
{
    bool issymmetric = true;
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<size_t> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            bool skip = false;

            std::vector<size_t> negindex(index.size());
            for(size_t idx = 0; idx < index.size(); ++idx)
            {
                if(index[idx] >= length[idx] / 2 + 1)
                {
                    skip = true;
                    break;
                }
                negindex[idx] = (length[idx] - index[idx]) % length[idx];
            }
            if(negindex[0] >= length[0] / 2 + 1)
            {
                skip = true;
            }

            if(!skip)
            {
                const auto i
                    = std::inner_product(index.begin(), index.end(), stride.begin(), b * dist);
                const auto j = std::inner_product(
                    negindex.begin(), negindex.end(), stride.begin(), b * dist);
                if(data[i] != std::conj(data[j]))
                {
                    std::cout << "(";
                    std::string separator;
                    for(auto val : index)
                    {
                        std::cout << separator << val;
                        separator = ",";
                    }
                    std::cout << ")->";
                    std::cout << i << "\t";
                    std::cout << "(";
                    separator = "";
                    for(auto val : negindex)
                    {
                        std::cout << separator << val;
                        separator = ",";
                    }
                    std::cout << ")->";
                    std::cout << j << ":\t";
                    std::cout << data[i] << " " << data[j];
                    std::cout << "\tnot conjugate!" << std::endl;
                    issymmetric = false;
                }
            }

        } while(increment_colmajor(index, length));
    }
    return issymmetric;
}

#endif
