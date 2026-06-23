/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dtype_cast.h
 * \brief Host-side FP16/BF16 conversion utilities.
 *
 * Provides software-emulated float-to-half/bfloat16 conversions for
 * host-side alpha/beta post-processing when hardware conversion is unavailable.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <securec.h>

namespace blas_common {

inline uint16_t FloatToHalf(float val)
{
    uint32_t f;
    memcpy_s(&f, sizeof(f), &val, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        uint32_t fp32Exp = (f >> 23) & 0xFF;
        uint32_t fp32Mant = f & 0x7FFFFF;
        if (fp32Exp == 0xFF && fp32Mant != 0) {
            return static_cast<uint16_t>(sign | 0x7E00); // NaN
        }
        return static_cast<uint16_t>(sign | 0x7C00); // Inf
    }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

inline float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (mant << 13);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy_s(&result, sizeof(result), &f, sizeof(f));
    return result;
}

inline uint16_t FloatToBf16(float val)
{
    uint32_t f;
    memcpy_s(&f, sizeof(f), &val, sizeof(val));
    uint32_t roundingBias = 0x7FFFU;
    if ((f & 0xFFFF) == 0x8000U) {
        roundingBias = 0x7FFEU;
    }
    uint32_t rounded = f + roundingBias;
    return static_cast<uint16_t>(rounded >> 16);
}

inline float Bf16ToFloat(uint16_t b)
{
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r;
    memcpy_s(&r, sizeof(r), &f, sizeof(f));
    return r;
}

} // namespace blas_common
