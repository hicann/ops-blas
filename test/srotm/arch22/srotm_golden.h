/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "types.h"

inline std::array<float, 5> getSrotmParams(const TestCaseConfig& tc)
{
    return {
        tc.sparam0.value_or(-1.0f),
        tc.sparam1.value_or(1.0f),
        tc.sparam2.value_or(0.0f),
        tc.sparam3.value_or(0.0f),
        tc.sparam4.value_or(1.0f),
    };
}

inline int64_t getSrotmStartIndex(int64_t n, int64_t inc)
{
    return inc >= 0 ? 0 : (1 - n) * inc;
}

inline void srotm_golden_impl(const TestCaseConfig& tc,
                              const std::vector<std::vector<float>>& inputs,
                              std::vector<float>& goldenX,
                              std::vector<float>& goldenY)
{
    const int64_t n = tc.n.value_or(0);
    const int64_t incx = tc.incx.value_or(1);
    const int64_t incy = tc.incy.value_or(1);
    const std::array<float, 5> sparam = getSrotmParams(tc);

    goldenX = inputs[0];
    goldenY = inputs[1];
    if (n <= 0 || sparam[0] == -2.0f) {
        return;
    }

    float h11 = 1.0f;
    float h12 = 0.0f;
    float h21 = 0.0f;
    float h22 = 1.0f;
    if (sparam[0] < 0.0f) {
        h11 = sparam[1];
        h21 = sparam[2];
        h12 = sparam[3];
        h22 = sparam[4];
    } else if (sparam[0] == 0.0f) {
        h12 = sparam[3];
        h21 = sparam[2];
    } else {
        h11 = sparam[1];
        h12 = 1.0f;
        h21 = -1.0f;
        h22 = sparam[4];
    }

    int64_t xIndex = getSrotmStartIndex(n, incx);
    int64_t yIndex = getSrotmStartIndex(n, incy);
    for (int64_t i = 0; i < n; ++i) {
        const auto xPos = static_cast<size_t>(xIndex);
        const auto yPos = static_cast<size_t>(yIndex);
        const float xValue = goldenX[xPos];
        const float yValue = goldenY[yPos];
        goldenX[xPos] = xValue * h11 + yValue * h12;
        goldenY[yPos] = xValue * h21 + yValue * h22;
        xIndex += incx;
        yIndex += incy;
    }
}
