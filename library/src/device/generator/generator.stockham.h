/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(_generator_stockham_H)
#define _generator_stockham_H
#include "rocfft.h"
#include <assert.h>
#include <map>
#include <sstream>
#include <stddef.h>
#include <string>
#include <vector>

/* =====================================================================
    Common small inline functions used in kernel generator
   =================================================================== */

typedef union
{
    float        f;
    unsigned int u;
    int          i;
} cb_t;

namespace StockhamGenerator
{

    // Convert unsigned floats to format string. std::to_string cannot format
    inline std::string FloatToStr(double f)
    {
        std::stringstream ss;
        ss.precision(16);
        ss << std::scientific << f;
        return ss.str();
    }

    //	Find the smallest power of 2 that is >= n; return its power of 2 factor
    //	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
    inline size_t CeilPo2(size_t n)
    {
        size_t v = 1, t = 0;
        while(v < n)
        {
            v <<= 1;
            t++;
        }

        return t;
    }

    inline size_t FloorPo2(size_t n)
    //	return the largest power of 2 that is <= n.
    //	e.g., FloorPo2 (7) returns 4.
    // *** TODO use x86 BSR instruction, using compiler intrinsics.
    {
        size_t tmp;
        while(0 != (tmp = n & (n - 1)))
            n = tmp;
        return n;
    }

    typedef std::pair<std::string, std::string> stringpair;
    inline stringpair
        ComplexMul(const char* type, const char* a, const char* b, bool forward = true)
    {
        stringpair result;
        result.first = "(";
        result.first += type;
        result.first += ") ((";
        result.first += a;
        result.first += ".x * ";
        result.first += b;
        result.first += (forward ? ".x - " : ".x + ");
        result.first += a;
        result.first += ".y * ";
        result.first += b;
        result.first += ".y),";
        result.second = "(";
        result.second += a;
        result.second += ".y * ";
        result.second += b;
        result.second += (forward ? ".x + " : ".x - ");
        result.second += a;
        result.second += ".x * ";
        result.second += b;
        result.second += ".y))";
        return result;
    }

    // Register data base types
    template <rocfft_precision PR>
    inline std::string RegBaseType(size_t count)
    {
        switch(count)
        {
        case 1:
            return "real_type_t<T>"; // float, double
        case 2:
            return "T"; // float2, double2
        case 4:
            return "vector4_type_t<T>"; // float4, double4
        default:
            assert(false);
            return "";
        }
    }

    template <rocfft_precision PR>
    inline std::string MakeRegBaseType(size_t count)
    {
        switch(count)
        {
        case 1:
            return "real_type_t<T>"; // float, double
        case 2:
            return "lib_make_vector2<T>"; // float2, double2
        case 4:
            return "lib_make_vector4< vector4_type_t<T> >"; // float4, double4
        default:
            assert(false);
            return "";
        }
    }

    template <rocfft_precision PR>
    inline std::string FloatSuffix()
    {
        // Suffix for constants
        std::string sfx;
        switch(PR)
        {
        case rocfft_precision_single:
            sfx = "f";
            break;
        case rocfft_precision_double:
            sfx = "";
            break;
        default:
            assert(false);
        }

        return sfx;
    }

    inline std::string ButterflyName(size_t radix, size_t count, bool fwd)
    {
        std::string str;
        if(fwd)
            str += "Fwd";
        else
            str += "Inv";
        str += "Rad";
        str += std::to_string(radix);
        str += "B";
        str += std::to_string(count);
        return str;
    }

    inline std::string PassName(size_t pos, bool fwd, size_t length, std::string name_suffix)
    {
        std::string str;
        if(fwd)
            str += "Fwd";
        else
            str += "Inv";
        str += "Pass";
        str += std::to_string(pos);
        str += "_len";
        str += std::to_string(length);
        str += name_suffix;
        return str;
    }

    inline std::string TwTableName()
    {
        return "twiddles";
    }

    inline std::string TwTableLargeName()
    {
        return "twiddle_dee";
    }

    inline std::string TwTableLargeFunc()
    {
        return "TW2step"; // TODO: switch according to the problem size
    }
};

#endif
