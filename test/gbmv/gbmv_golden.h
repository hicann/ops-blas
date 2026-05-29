/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBMV_CPU_H
#define GBMV_CPU_H

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// reference implementation — same signature as aclblasSgbmv
inline aclblasStatus_t aclblasSgbmv_cpu(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const float* alpha,
    const float* a, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || kl < 0 || ku < 0 || lda < kl + ku + 1)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;

    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);

    // beta * y
    if (beta != nullptr) {
        if (incy >= 0) {
            for (int i = 0; i < yCount; i++) y[i * incy] *= (*beta);
        } else {
            for (int i = 0; i < yCount; i++) y[(yCount - 1 - i) * absIncy] *= (*beta);
        }
    }

    if (yCount == 0) return ACLBLAS_STATUS_SUCCESS;

    if (isTransN) {
        // y = alpha * A * x + beta*y
        for (int iRow = 0; iRow < m; iRow++) {
            double sum = 0.0;
            int jStart = std::max(0, iRow - static_cast<int>(kl));
            int jEnd   = std::min(n - 1, iRow + static_cast<int>(ku));
            for (int j = jStart; j <= jEnd; j++) {
                int bandIdx = (ku + iRow - j) + j * lda;
                const float* xPtr = (incx >= 0) ? (x + j * incx)
                                                 : (x + (xCount - 1 - j) * absIncx);
                sum += static_cast<double>(a[bandIdx]) * static_cast<double>(*xPtr);
            }
            int yIdx = (incy >= 0) ? (iRow * incy) : ((yCount - 1 - iRow) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(*alpha) * sum +
                       static_cast<double>(y[yIdx]));
        }
    } else {
        // y = alpha * A^T * x + beta*y  (or A^H, same for real)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            int iStart = std::max(0, j - static_cast<int>(ku));
            int iEnd   = std::min(m - 1, j + static_cast<int>(kl));
            for (int i = iStart; i <= iEnd; i++) {
                int bandIdx = (ku + i - j) + j * lda;
                const float* xPtr = (incx >= 0) ? (x + i * incx)
                                                 : (x + (xCount - 1 - i) * absIncx);
                sum += static_cast<double>(a[bandIdx]) * static_cast<double>(*xPtr);
            }
            int yIdx = (incy >= 0) ? (j * incy) : ((yCount - 1 - j) * absIncy);
            y[yIdx] = static_cast<float>(static_cast<double>(*alpha) * sum +
                       static_cast<double>(y[yIdx]));
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GBMV_CPU_H
