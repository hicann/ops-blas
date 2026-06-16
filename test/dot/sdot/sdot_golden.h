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

#include <cstdint>
#include <vector>

inline float aclblasSdot_cpu(int n, const float* x, int incx, const float* y, int incy)
{
    if (n <= 0) {
        return 0.0f;
    }

    int absIncx = std::abs(incx);
    int absIncy = std::abs(incy);

    const float* xStart = (incx < 0) ? (x + (n - 1) * absIncx) : x;
    const float* yStart = (incy < 0) ? (y + (n - 1) * absIncy) : y;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += static_cast<double>(xStart[i * incx]) * static_cast<double>(yStart[i * incy]);
    }
    return static_cast<float>(sum);
}

