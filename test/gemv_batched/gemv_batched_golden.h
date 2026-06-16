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

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t validateGemvBatchedCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, int lda, int incx,
    const float* beta, int incy, int batchCount)
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
    if (batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

// Per-batch GEMV: y = alpha * op(A) * x + beta * y
// For real types OP_C (conjugate-transpose) is equivalent to OP_T.
static void gemvBatchedCpuOne(
    aclblasOperation_t trans, int m, int n, float alpha, const float* a, int lda,
    const float* x, int incx, float beta, float* y, int incy)
{
    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);

    for (int i = 0; i < yCount; i++) {
        int yIdx = (incy > 0) ? (i * incy) : ((yCount - 1 - i) * absIncy);
        y[yIdx] *= beta;
    }

    if (isTransN) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                int xIdx = (incx > 0) ? (j * incx) : ((xCount - 1 - j) * absIncx);
                sum += static_cast<double>(a[i + static_cast<int64_t>(j) * lda]) * static_cast<double>(x[xIdx]);
            }
            int yIdx = (incy > 0) ? (i * incy) : ((yCount - 1 - i) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(alpha) * sum + static_cast<double>(y[yIdx]));
        }
    } else {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                int xIdx = (incx > 0) ? (i * incx) : ((xCount - 1 - i) * absIncx);
                sum += static_cast<double>(a[i + static_cast<int64_t>(j) * lda]) * static_cast<double>(x[xIdx]);
            }
            int yIdx = (incy > 0) ? (j * incy) : ((yCount - 1 - j) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(alpha) * sum + static_cast<double>(y[yIdx]));
        }
    }
}

// Unified CPU golden: computes in float using caller-quantized inputs
inline aclblasStatus_t aclblasGemvBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    aclblasStatus_t validRet = validateGemvBatchedCpuParams(handle, trans, m, n, alpha, lda, incx, beta, incy, batchCount);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (m == 0 || n == 0 || batchCount == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    const size_t xStride = static_cast<size_t>((xCount - 1) * std::abs(incx) + 1);
    const size_t yStride = static_cast<size_t>((yCount - 1) * std::abs(incy) + 1);
    const size_t aStride = static_cast<size_t>(lda) * n;

    for (int b = 0; b < batchCount; b++) {
        gemvBatchedCpuOne(trans, m, n, *alpha, a + b * aStride, lda, x + b * xStride, incx, *beta, y + b * yStride, incy);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

