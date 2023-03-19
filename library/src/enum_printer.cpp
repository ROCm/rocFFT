// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "enum_printer.h"

#include <map>
#include <set>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)
#define STRENUM(x) TO_STR(x), x

static std::map<rocfft_precision, const char*> PrecisionToStrMap()
{
    std::map<rocfft_precision, const char*> PrecisionToStr = {{rocfft_precision_single, "single"},
                                                              {rocfft_precision_double, "double"},
                                                              {rocfft_precision_half, "half"}};
    return PrecisionToStr;
}

static std::map<std::string, rocfft_precision> StrToPrecisionMap()
{
    std::map<std::string, rocfft_precision> StrToPrecision;
    for(auto i : PrecisionToStrMap())
        StrToPrecision.emplace(i.second, i.first);
    return StrToPrecision;
}

static std::map<SBRC_TRANSPOSE_TYPE, const char*> SBRCTransTypetoStrMap()
{
    std::map<SBRC_TRANSPOSE_TYPE, const char*> SBRCTransTypeToStr = {
        {ENUMSTR(NONE)}, {ENUMSTR(DIAGONAL)}, {ENUMSTR(TILE_ALIGNED)}, {ENUMSTR(TILE_UNALIGNED)}};
    return SBRCTransTypeToStr;
}

static std::map<std::string, SBRC_TRANSPOSE_TYPE> StrToSBRCTransTypeMap()
{
    std::map<std::string, SBRC_TRANSPOSE_TYPE> StrToSBRCTransType;
    for(auto i : SBRCTransTypetoStrMap())
        StrToSBRCTransType.emplace(i.second, i.first);
    return StrToSBRCTransType;
}

std::string PrintOperatingBuffer(const OperatingBuffer ob)
{
    static const std::map<OperatingBuffer, const char*> BuffertoString
        = {{ENUMSTR(OB_UNINIT)},
           {ENUMSTR(OB_USER_IN)},
           {ENUMSTR(OB_USER_OUT)},
           {ENUMSTR(OB_TEMP)},
           {ENUMSTR(OB_TEMP_CMPLX_FOR_REAL)},
           {ENUMSTR(OB_TEMP_BLUESTEIN)}};
    return BuffertoString.at(ob);
}

std::string PrintOperatingBufferCode(const OperatingBuffer ob)
{
    static const std::map<OperatingBuffer, const char*> BuffertoString
        = {{OB_UNINIT, "ERR"},
           {OB_USER_IN, "A"},
           {OB_USER_OUT, "B"},
           {OB_TEMP, "T"},
           {OB_TEMP_CMPLX_FOR_REAL, "C"},
           {OB_TEMP_BLUESTEIN, "S"}};
    return BuffertoString.at(ob);
}

std::string PrintOptimizeStrategy(const rocfft_optimize_strategy ros)
{
    static const std::map<rocfft_optimize_strategy, const char*> StrategytoString
        = {{rocfft_optimize_min_buffer, "MINIMIZE_BUFFER"},
           {rocfft_optimize_balance, "BALANCE_BUFFER_FUSION"},
           {rocfft_optimize_max_fusion, "MAXIMIZE_FUSION"}};
    return StrategytoString.at(ros);
}

std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty)
{
    static auto sbrc2strMap = SBRCTransTypetoStrMap();
    return sbrc2strMap.at(ty);
}

std::string PrintDirectToFromRegMode(const DirectRegType ty)
{
    static const std::map<DirectRegType, const char*> TypetoString
        = {{ENUMSTR(FORCE_OFF_OR_NOT_SUPPORT)}, {ENUMSTR(TRY_ENABLE_IF_SUPPORT)}};
    return TypetoString.at(ty);
}

std::string PrintPrecision(const rocfft_precision pre)
{
    static auto precision2strMap = PrecisionToStrMap();
    return precision2strMap.at(pre);
}

std::string PrintArrayType(const rocfft_array_type aryType)
{
    static const std::map<rocfft_array_type, const char*> aryTypeStr
        = {{rocfft_array_type_complex_interleaved, "CI"},
           {rocfft_array_type_complex_planar, "CP"},
           {rocfft_array_type_real, "R"},
           {rocfft_array_type_hermitian_interleaved, "HI"},
           {rocfft_array_type_hermitian_planar, "HP"},
           {rocfft_array_type_unset, "NA"}};
    return aryTypeStr.at(aryType);
}
std::string PrintPlacement(const rocfft_result_placement placement)
{
    static const std::map<rocfft_result_placement, const char*> placementStr
        = {{rocfft_placement_inplace, "IP"}, {rocfft_placement_notinplace, "OP"}};
    return placementStr.at(placement);
}
std::string PrintEBType(const EmbeddedType ebtype)
{
    if(ebtype == EmbeddedType::NONE)
        return std::string("NONE");
    else if(ebtype == EmbeddedType::Real2C_POST)
        return std::string("POST");
    else
        return std::string("PRE");
}

SBRC_TRANSPOSE_TYPE StrToSBRCTransType(const std::string& str)
{
    static auto str2sbrcMap = StrToSBRCTransTypeMap();
    return str2sbrcMap.at(str);
}

rocfft_precision StrToPrecision(const std::string& str)
{
    static auto str2precisionMap = StrToPrecisionMap();
    return str2precisionMap.at(str);
}