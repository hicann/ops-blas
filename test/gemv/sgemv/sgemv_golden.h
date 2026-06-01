/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMV_SGEMV_GOLDEN_H
#define GEMV_SGEMV_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t validateSgemvCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, int lda, int incx,
    const float* beta, int incy)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSgemv_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    aclblasStatus_t validRet = validateSgemvCpuParams(handle, trans, m, n, alpha, lda, incx, beta, incy);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);

    // y = beta * y
    for (int i = 0; i < yCount; i++) {
        int yIdx = (incy > 0) ? (i * incy) : ((yCount - 1 - i) * absIncy);
        y[yIdx] *= (*beta);
    }

    if (isTransN) {
        // y = alpha * A * x + y
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                int xIdx = (incx > 0) ? (j * incx) : ((xCount - 1 - j) * absIncx);
                sum += static_cast<double>(a[i + static_cast<int64_t>(j) * lda]) * static_cast<double>(x[xIdx]);
            }
            int yIdx = (incy > 0) ? (i * incy) : ((yCount - 1 - i) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(*alpha) * sum + static_cast<double>(y[yIdx]));
        }
    } else {
        // y = alpha * A^T * x + y  (or A^H, same for real)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                int xIdx = (incx > 0) ? (i * incx) : ((xCount - 1 - i) * absIncx);
                sum += static_cast<double>(a[i + static_cast<int64_t>(j) * lda]) * static_cast<double>(x[xIdx]);
            }
            int yIdx = (incy > 0) ? (j * incy) : ((yCount - 1 - j) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(*alpha) * sum + static_cast<double>(y[yIdx]));
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GEMV_SGEMV_GOLDEN_H
