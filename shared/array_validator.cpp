// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <numeric>
#include <unordered_set>

#include "array_validator.h"
#include "increment.h"

// Check a 2D array for collisions.
// The 2D case can be determined via a number-theoretic argument.
bool valid_length_stride_2d(const size_t l0, const size_t l1, const size_t s0, const size_t s1)
{
    if(s0 == s1)
        return false;
    const auto c = std::lcm(s0, s1);
    return !((s0 * (l0 - 1) >= c) && (s1 * (l1 - 1) >= c));
}

// Compare a 1D direction with a multi-index hyperface for collisions.
bool valid_length_stride_1d_multi(const unsigned int        idx,
                                  const std::vector<size_t> l,
                                  const std::vector<size_t> s,
                                  const int                 verbose)
{
    size_t              l0, s0;
    std::vector<size_t> l1{}, s1{};
    for(unsigned int i = 0; i < l.size(); ++i)
    {
        if(i == idx)
        {
            l0 = l[i];
            s0 = s[i];
        }
        else
        {
            l1.push_back(l[i]);
            s1.push_back(s[i]);
        }
    }

    if(verbose > 4)
        std::cout << "l0: " << l0 << "\ts0: " << s0 << std::endl;

    // We only need to go to the maximum pointer offset for (l1,s1).
    const auto max_offset
        = std::accumulate(l1.begin(), l1.end(), (size_t)1, std::multiplies<size_t>())
          - std ::inner_product(l1.begin(), l1.end(), s1.begin(), (size_t)0);
    std::unordered_set<size_t> a0{};
    for(size_t i = 1; i < l0; ++i)
    {
        const auto val = i * s0;
        if(val <= max_offset)
            a0.insert(val);
        else
            break;
    }

    if(verbose > 5)
    {
        std::cout << "a0:";
        for(auto i : a0)
            std::cout << " " << i;
        std::cout << std::endl;

        std::cout << "l1:";
        for(auto i : l1)
            std::cout << " " << i;
        std::cout << std::endl;

        std::cout << "s1:";
        for(auto i : s1)
            std::cout << " " << i;
        std::cout << std::endl;
    }

    // TODO: this can be multi-threaded, since find(...) is thread-safe.
    std::vector<size_t> index(l1.size());
    std::fill(index.begin(), index.end(), 0);
    do
    {
        const int i = std::inner_product(index.begin(), index.end(), s1.begin(), (size_t)0);
        if(i > 0)
        {
            if(verbose > 6)
                std::cout << i << std::endl;
            if(a0.find(i) != a0.end())
                return false;
        }
    } while(increment_rowmajor(index, l1));

    return true;
}

// Compare a hyperface with another hyperface for collisions.
bool valid_length_stride_multi_multi(const std::vector<size_t> l0,
                                     const std::vector<size_t> s0,
                                     const std::vector<size_t> l1,
                                     const std::vector<size_t> s1)
{
    std::unordered_set<size_t> a0{};

    const auto max_offset
        = std::accumulate(l1.begin(), l1.end(), (size_t)1, std::multiplies<size_t>())
          - std::inner_product(l1.begin(), l1.end(), s1.begin(), (size_t)0);
    std::vector<size_t> index0(l0.size());
    std::fill(index0.begin(), index0.end(), 0);
    do
    {
        const auto i = std::inner_product(index0.begin(), index0.end(), s0.begin(), (size_t)0);
        if(i > max_offset)
            a0.insert(i);
    } while(increment_rowmajor(index0, l0));

    std::vector<size_t> index1(l1.size());
    std::fill(index1.begin(), index1.end(), 0);
    do
    {
        const auto i = std::inner_product(index1.begin(), index1.end(), s1.begin(), (size_t)0);
        if(i > 0)
        {
            if(a0.find(i) != a0.end())
            {

                return false;
            }
        }
    } while(increment_rowmajor(index1, l1));

    return true;
}

bool valid_length_stride_3d(const std::vector<size_t>& l,
                            const std::vector<size_t>& s,
                            const int                  verbose)
{
    // Check that 2D faces are valid:
    if(!valid_length_stride_2d(l[0], l[1], s[0], s[1]))
        return false;
    if(!valid_length_stride_2d(l[0], l[2], s[0], s[2]))
        return false;
    if(!valid_length_stride_2d(l[1], l[2], s[1], s[2]))
        return false;

    // If the 2D faces are valid, check an axis vs a face for collisions:
    if(!valid_length_stride_1d_multi(0, l, s, verbose))
        return false;
    if(!valid_length_stride_1d_multi(1, l, s, verbose))
        return false;
    if(!valid_length_stride_1d_multi(2, l, s, verbose))
        return false;

    return true;
}

bool valid_length_stride_4d(const std::vector<size_t>& l,
                            const std::vector<size_t>& s,
                            const int                  verbose)
{
    if(l.size() != 4)
    {
        throw std::runtime_error("Incorrect dimensions for valid_length_stride_4d");
    }

    // Check that 2D faces are valid:
    for(int idx0 = 0; idx0 < 3; ++idx0)
    {
        for(int idx1 = idx0 + 1; idx1 < 4; ++idx1)
        {
            if(!valid_length_stride_2d(l[idx0], l[idx1], s[idx0], s[idx1]))
                return false;
        }
    }

    // Check that 1D vs 3D faces are valid:
    for(int idx0 = 0; idx0 < 4; ++idx0)
    {
        if(!valid_length_stride_1d_multi(idx0, l, s, verbose))
            return false;
    }

    // Check that 2D vs 2D faces are valid:
    for(int idx0 = 0; idx0 < 3; ++idx0)
    {
        for(int idx1 = idx0 + 1; idx1 < 4; ++idx1)
        {
            int idx2 = -1;
            for(int i = 0; i < 4; ++i)
            {
                if(i != idx0 && i != idx1)
                {
                    idx2 = i;
                    break;
                }
            }
            int idx3 = -1;
            for(int i = 0; i < 4; ++i)
            {
                if(i != idx0 && i != idx1 && i != idx2)
                {
                    idx3 = i;
                    break;
                }
            }
            std::vector<size_t> l0{l[idx0], l[idx1]};
            std::vector<size_t> s0{s[idx0], s[idx1]};

            std::vector<size_t> l1{l[idx2], l[idx3]};
            std::vector<size_t> s1{s[idx2], s[idx3]};

            if(!valid_length_stride_multi_multi(l0, s0, l1, s1))
                return false;
        }
    }

    return true;
}

bool valid_length_stride_generald(const std::vector<size_t> l,
                                  const std::vector<size_t> s,
                                  const int                 verbose)
{
    if(verbose > 2)
    {
        std::cout << l.size() << std::endl;
    }

    // Recurse on d-1 hyper-faces:
    for(unsigned int idx = 0; idx < l.size(); ++idx)
    {
        std::vector<size_t> l0{};
        std::vector<size_t> s0{};
        for(size_t i = 0; i < l.size(); ++i)
        {
            if(i != idx)
            {
                l0.push_back(l[i]);
                s0.push_back(s[i]);
            }
        }
        if(!array_valid(l0, s0, verbose))
            return false;
    }

    // Handle the 1D vs (N-1) case:
    for(unsigned int idx = 0; idx < l.size(); ++idx)
    {
        if(!valid_length_stride_1d_multi(idx, l, s, verbose))
            return false;
    }

    for(size_t dim0 = 2; dim0 <= l.size() / 2; ++dim0)
    {
        const size_t dim1 = l.size() - dim0;
        if(verbose > 2)
            std::cout << "dims: " << dim0 << " " << dim1 << std::endl;

        std::vector<size_t> l0;
        std::vector<size_t> s0;
        std::vector<size_t> l1;
        std::vector<size_t> s1;
        l0.reserve(dim0);
        s0.reserve(dim0);
        l1.reserve(dim1);
        s1.reserve(dim1);

        // We iterate over all permutations of an array of length l.size() which contains dim0 zeros
        // and dim1 ones.  We start with {0, ..., 0, 1, ... 1} to guarantee that we hit all the
        // possibilities.

        std::vector<size_t> v(l.size());
        std::fill(v.begin(), v.begin() + dim1, 0);
        std::fill(v.begin() + dim1, v.end(), 1);
        do
        {
            if(verbose > 3)
            {
                std::cout << "v:";
                for(const auto i : v)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
            }

            l0.clear();
            s0.clear();
            l1.clear();
            s1.clear();

            for(size_t i = 0; i < l.size(); ++i)
            {
                if(v[i] == 0)
                {
                    l0.push_back(l[i]);
                    s0.push_back(s[i]);
                }
                else
                {
                    l1.push_back(l[i]);
                    s1.push_back(s[i]);
                }
            }

            if(verbose > 3)
            {
                std::cout << "\tl0:";
                for(const auto i : l0)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\ts0:";
                for(const auto i : s0)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\tl1:";
                for(const auto i : l1)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\ts1:";
                for(const auto i : s1)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
            }

            if(!valid_length_stride_multi_multi(l0, s0, l1, s1))
            {
                return false;
            }

        } while(std::next_permutation(v.begin(), v.end()));
    }

    return true;
}

bool array_valid(const std::vector<size_t>& length,
                 const std::vector<size_t>& stride,
                 const int                  verbose)
{
    if(length.size() != stride.size())
        return false;

    // If a length is 1, then the stride is irrelevant.
    // If a length is > 1, then the corresponding stride must be > 1.
    std::vector<size_t> l{}, s{};
    for(unsigned int i = 0; i < length.size(); ++i)
    {
        if(length[i] > 1)
        {
            if(stride[i] == 0)
                return false;
            l.push_back(length[i]);
            s.push_back(stride[i]);
        }
    }

    switch(l.size())
    {
    case 0:
        return true;
        break;
    case 1:
        return s[0] != 0;
        break;
    case 2:
    {
        return valid_length_stride_2d(l[0], l[1], s[0], s[1]);
        break;
    }
    case 3:
    {
        return valid_length_stride_3d(l, s, verbose);
        break;
    }
    case 4:
    {
        return valid_length_stride_4d(l, s, verbose);
        break;
    }
    default:
        return valid_length_stride_generald(l, s, verbose);
        return true;
    }

    return true;
}
