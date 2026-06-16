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
#include <cstdint>
#include <climits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// CPU reference implementation for aclblasSger (BLAS Level 2 rank-1 update).
// Column-major storage: A[i][j] = A[i + j * lda].
// Computation: A := alpha * x * y^T + A.

// Validate parameters for cpu reference, mirrors host-side aclblasSger.
// Returns error code or SUCCESS; sets shouldCompute=true when computation should proceed.
inline aclblasStatus_t ValidateSgerParamsCpu(
    aclblasHandle_t handle, int m, int n, int lda, int incx, int incy,
    const float* alpha, const float* x, const float* y, const float* A, bool& shouldCompute)
{
    shouldCompute = false;
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m < 0 || n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == INT_MIN || incy == INT_MIN) return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m > 0 && n > 0) {
        if (x == nullptr || y == nullptr || A == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;
    shouldCompute = true;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSger_cpu(
    aclblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    bool shouldCompute = false;
    aclblasStatus_t st = ValidateSgerParamsCpu(handle, m, n, lda, incx, incy, alpha, x, y, A, shouldCompute);
    if (!shouldCompute) return st;

    // A := alpha * x * y^T + A  (column-major storage)
    float alphaVal = *alpha;
    for (int64_t i = 0; i < m; i++) {
        int64_t xIdx = (incx >= 0)
            ? i * static_cast<int64_t>(incx)
            : (m - 1 - i) * static_cast<int64_t>(-incx);
        float xi = x[xIdx];
        for (int64_t j = 0; j < n; j++) {
            int64_t yIdx = (incy >= 0)
                ? j * static_cast<int64_t>(incy)
                : (n - 1 - j) * static_cast<int64_t>(-incy);
            int64_t aIdx = i + j * static_cast<int64_t>(lda);
            A[aIdx] += alphaVal * xi * y[yIdx];
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

