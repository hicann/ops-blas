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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline void aclblasSsyr2_cpu(
    aclblasFillMode_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)
{
    if (n <= 0)
        return;

    int absIncx = std::abs(incx);
    int absIncy = std::abs(incy);

    auto getX = [&](int i) -> float { return (incx >= 0) ? x[i * incx] : x[(n - 1 - i) * absIncx]; };
    auto getY = [&](int i) -> float { return (incy >= 0) ? y[i * incy] : y[(n - 1 - i) * absIncy]; };

    for (int j = 0; j < n; j++) {
        float xj = getX(j);
        float yj = getY(j);
        float axj = alpha * xj;
        float ayj = alpha * yj;
        if (uplo == ACLBLAS_UPPER) {
            for (int i = 0; i <= j; i++) {
                A[j * lda + i] += axj * getY(i) + ayj * getX(i);
            }
        } else {
            for (int i = j; i < n; i++) {
                A[j * lda + i] += axj * getY(i) + ayj * getX(i);
            }
        }
    }
}

