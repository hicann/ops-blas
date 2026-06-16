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

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasSrotm_cpu(
    aclblasHandle_t handle,
    int64_t n, float* x, int64_t incx,
    float* y, int64_t incy, const float* sparam)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (x == nullptr || y == nullptr || sparam == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;

    if (n <= 0 || sparam[0] == -2.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    float h11 = 1.0f, h12 = 0.0f, h21 = 0.0f, h22 = 1.0f;
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

    int64_t xStartIndex = (incx >= 0) ? 0 : (1 - n) * incx;
    int64_t yStartIndex = (incy >= 0) ? 0 : (1 - n) * incy;

    for (int64_t i = 0; i < n; ++i) {
        int64_t xIdx = xStartIndex + i * incx;
        int64_t yIdx = yStartIndex + i * incy;
        float xVal = x[xIdx];
        float yVal = y[yIdx];
        x[xIdx] = xVal * h11 + yVal * h12;
        y[yIdx] = xVal * h21 + yVal * h22;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

