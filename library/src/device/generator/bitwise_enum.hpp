/*******************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#include <type_traits>

template <typename TEnumType>
struct support_bitwise_enum : std::false_type
{
};

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator&(TEnumType left, TEnumType right)
{
    return static_cast<TEnumType>(static_cast<std::underlying_type_t<TEnumType>>(left)
                                  & static_cast<std::underlying_type_t<TEnumType>>(right));
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator|(TEnumType left, TEnumType right)
{
    return static_cast<TEnumType>(static_cast<std::underlying_type_t<TEnumType>>(left)
                                  | static_cast<std::underlying_type_t<TEnumType>>(right));
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator^(TEnumType left, TEnumType right)
{
    return static_cast<TEnumType>(static_cast<std::underlying_type_t<TEnumType>>(left)
                                  ^ static_cast<std::underlying_type_t<TEnumType>>(right));
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator~(TEnumType value)
{
    return static_cast<TEnumType>(~static_cast<std::underlying_type_t<TEnumType>>(value));
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator&=(TEnumType& left, TEnumType right)
{
    left = left & right;
    return left;
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator|=(TEnumType& left, TEnumType right)
{
    left = left | right;
    return left;
}

template <typename TEnumType>
typename std::enable_if_t<support_bitwise_enum<TEnumType>::value, TEnumType>
    operator^=(TEnumType& left, TEnumType right)
{
    left = left ^ right;
    return left;
}