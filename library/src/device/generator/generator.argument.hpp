/*******************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#include "bitwise_enum.hpp"

enum class EPrecision : uint8_t
{
    NONE   = 0b00,
    SINGLE = 0b01,
    DOUBLE = 0b10,
    ALL    = SINGLE | DOUBLE
};

enum class EPredefineType : uint8_t
{
    NONE  = 0b0000000,
    POW2  = 0b0000001,
    POW3  = 0b0000010,
    POW5  = 0b0000100,
    POW7  = 0b0001000, // actually is 7,11,13
    SMALL = POW2 | POW3 | POW5 | POW7,
    LARGE = 0b0010000,
    DIM2  = 0b0100000,
    ALL   = SMALL | LARGE | DIM2,
};

template <>
struct support_bitwise_enum<EPrecision> : std::true_type
{
};
template <>
struct support_bitwise_enum<EPredefineType> : std::true_type
{
};

class generator_argument
{
public:
    size_t              group_num     = 8;
    EPrecision          precision     = EPrecision::ALL;
    EPredefineType      predefineType = EPredefineType::ALL;
    std::vector<size_t> manualSize;
    std::vector<size_t> manualSizeLarge;
    std::set<size_t>    validManualSize;
    std::set<size_t>    validManualSizeLarge;

    void init_precision(const std::vector<std::string>& argString)
    {
        // we're here only when -p is in the args, starting from none and do bit-OR
        precision = EPrecision::NONE;
        if(std::find(argString.begin(), argString.end(), "single") != argString.end())
        {
            precision |= EPrecision::SINGLE;
        }
        if(std::find(argString.begin(), argString.end(), "double") != argString.end())
        {
            precision |= EPrecision::DOUBLE;
        }
        if(std::find(argString.begin(), argString.end(), "all") != argString.end())
        {
            precision |= EPrecision::ALL;
        }
    }

    void init_type(const std::vector<std::string>& argString)
    {
        // we're here only when -t is in the args, starting from none and do bit-OR
        predefineType = EPredefineType::NONE;
        if(std::find(argString.begin(), argString.end(), "pow2") != argString.end())
        {
            predefineType |= EPredefineType::POW2;
        }
        if(std::find(argString.begin(), argString.end(), "pow3") != argString.end())
        {
            predefineType |= EPredefineType::POW3;
        }
        if(std::find(argString.begin(), argString.end(), "pow5") != argString.end())
        {
            predefineType |= EPredefineType::POW5;
        }
        if(std::find(argString.begin(), argString.end(), "pow7") != argString.end())
        {
            predefineType |= EPredefineType::POW7;
        }
        if(std::find(argString.begin(), argString.end(), "small") != argString.end())
        {
            predefineType |= EPredefineType::SMALL;
        }
        if(std::find(argString.begin(), argString.end(), "large") != argString.end())
        {
            predefineType |= EPredefineType::LARGE;
        }
        if(std::find(argString.begin(), argString.end(), "2D") != argString.end())
        {
            predefineType |= EPredefineType::DIM2;
        }
        if(std::find(argString.begin(), argString.end(), "all") != argString.end())
        {
            predefineType |= EPredefineType::ALL;
        }
    }

    size_t filter_manual_small_size(std::set<size_t>& pool)
    {
        for(auto i : manualSize)
        {
            // get the valid size
            if(pool.count(i))
                validManualSize.insert(i);
        }
        // return the final number of valid sizes
        return validManualSize.size();
    }

    size_t filter_manual_large_size(std::set<size_t>& pool)
    {
        for(auto i : manualSizeLarge)
        {
            // get the valid size
            if(pool.count(i))
                validManualSizeLarge.insert(i);
        }
        // return the final number of valid sizes
        return validManualSizeLarge.size();
    }

    bool has_precision(EPrecision testPrecision) const
    {
        return ((precision & testPrecision) != EPrecision::NONE);
    }

    bool has_predefine_type(EPredefineType testType) const
    {
        return ((predefineType & testType) != EPredefineType::NONE);
    }

    bool has_manual_small_size() const
    {
        return validManualSize.size() > 0;
    }

    bool has_manual_large_size() const
    {
        return validManualSizeLarge.size() > 0;
    }

    bool check_valid() const
    {
        // no valid precision
        if(precision == EPrecision::NONE)
        {
            std::cerr << "No valid precision!" << std::endl;
            return false;
        }

        // no any size to gen
        if(predefineType == EPredefineType::NONE && !has_manual_small_size()
           && !has_manual_large_size())
        {
            std::cerr << "No valid sizes to generate!" << std::endl;
            return false;
        }

        return true;
    }

    // Convert to string for output.
    std::string str(const std::string& separator = "\n") const
    {
        std::stringstream ss;

        ss << "type:";
        if(predefineType == EPredefineType::NONE)
            ss << " none";
        if(has_predefine_type(EPredefineType::POW2))
            ss << " pow2";
        if(has_predefine_type(EPredefineType::POW3))
            ss << " pow3";
        if(has_predefine_type(EPredefineType::POW5))
            ss << " pow5";
        if(has_predefine_type(EPredefineType::POW7))
            ss << " pow7,11,13";
        if(has_predefine_type(EPredefineType::LARGE))
            ss << " large";
        if(has_predefine_type(EPredefineType::DIM2))
            ss << " 2D";
        ss << separator;

        ss << "valid manual small size:";
        for(auto i : validManualSize)
            ss << " " << i;
        ss << separator;

        ss << "valid manual large size:";
        for(auto i : validManualSizeLarge)
            ss << " " << i;
        ss << separator;

        ss << "precision:";
        if(has_precision(EPrecision::SINGLE))
            ss << " single";
        if(has_precision(EPrecision::DOUBLE))
            ss << " double";
        ss << separator;

        ss << "group_num: " << group_num << separator;

        return ss.str();
    }
};