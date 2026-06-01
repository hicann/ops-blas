/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRMV_CPU_H
#define TRMV_CPU_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline size_t StrmvVectorIndex(int idx, int n, int incx)
{
    if (incx >= 0) {
        return static_cast<size_t>(static_cast<int64_t>(idx) * incx);
    }
    return static_cast<size_t>(static_cast<int64_t>(n - 1 - idx) * (-static_cast<int64_t>(incx)));
}

inline aclblasStatus_t StrmvValidateParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, const float* a, int lda, const float* x, int incx)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < n || incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (a == nullptr || x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* a, int lda, float* x, int incx)
{
    aclblasStatus_t valRet = StrmvValidateParams(handle, uplo, trans, diag, n, a, lda, x, incx);
    if (valRet != ACLBLAS_STATUS_SUCCESS) return valRet;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    const int absIncx = std::abs(incx);
    const size_t xLen = static_cast<size_t>(absIncx * (n - 1) + 1);
    std::vector<float> xCopy(x, x + xLen);

    for (int row = 0; row < n; ++row) {
        float acc = 0.0f;
        int colStart = 0;
        int colEnd = n;
        const bool transIsN = (trans == ACLBLAS_OP_N);
        if (transIsN) {
            if (uplo == ACLBLAS_UPPER) {
                colStart = row;
            } else {
                colEnd = row + 1;
            }
        } else {
            if (uplo == ACLBLAS_UPPER) {
                colEnd = row + 1;
            } else {
                colStart = row;
            }
        }
        for (int col = colStart; col < colEnd; ++col) {
            float aVal = 1.0f;
            if (diag != ACLBLAS_UNIT || col != row) {
                aVal = transIsN ? a[static_cast<size_t>(row + lda * col)]
                                : a[static_cast<size_t>(col + lda * row)];
            }
            acc += aVal * xCopy[StrmvVectorIndex(col, n, incx)];
        }
        x[StrmvVectorIndex(row, n, incx)] = acc;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // TRMV_CPU_H
