/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file host_utils.h
 * \brief Host-side utilities.
 */

#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <cstdint>
#include <limits>
#include <type_traits>

// ==========================================================================
//  CeilDiv — integer ceiling division:  ceil(a / b)
// ==========================================================================

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilDiv(T1 a, T2 b)
{
    static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
                  "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_arithmetic<R>::value,
                  "CeilDiv return type must be arithmetic");
    auto ra = static_cast<R>(a);
    auto rb = static_cast<R>(b);
    if (rb == 0) {
        return std::numeric_limits<R>::max();
    }
    // Overflow guard: if a + b - 1 wraps around, return max
    if (ra + rb - 1 < ra) {
        return std::numeric_limits<R>::max();
    }
    return (ra + rb - 1) / rb;
}

// ==========================================================================
//  CeilAlign — round val up to a multiple of align:  CeilDiv(val,align)*align
// ==========================================================================

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}

#endif  // HOST_UTILS_H
