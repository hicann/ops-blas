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
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline size_t StbmvVectorIndex(int idx, int n, int incx)
{
    if (incx >= 0) {
        return static_cast<size_t>(static_cast<int64_t>(idx) * incx);
    }
    return static_cast<size_t>(static_cast<int64_t>(n - 1 - idx) * (-static_cast<int64_t>(incx)));
}

inline size_t StbmvBandIndex(aclblasFillMode_t uplo, int row, int col, int k, int lda)
{
    if (uplo == ACLBLAS_UPPER) {
        return static_cast<size_t>(
            static_cast<int64_t>(k + row - col) + static_cast<int64_t>(col) * static_cast<int64_t>(lda));
    }
    return static_cast<size_t>(
        static_cast<int64_t>(row - col) + static_cast<int64_t>(col) * static_cast<int64_t>(lda));
}

inline bool StbmvIsStored(aclblasFillMode_t uplo, int row, int col, int n, int k)
{
    if (uplo == ACLBLAS_UPPER) {
        return row >= std::max(0, col - k) && row <= col;
    }
    return row >= col && row <= std::min(n - 1, col + k);
}

inline float StbmvGetBandValue(
    const float* a, aclblasFillMode_t uplo, aclblasDiagType_t diag, int row, int col, int n, int k,
    int lda)
{
    if (row == col && diag == ACLBLAS_UNIT) {
        return 1.0f;
    }
    if (!StbmvIsStored(uplo, row, col, n, k)) {
        return 0.0f;
    }
    return a[StbmvBandIndex(uplo, row, col, k, lda)];
}

inline aclblasStatus_t StbmvValidateParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* a, int lda, const float* x, int incx)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || k < 0 || lda < k + 1 || incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (a == nullptr || x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStbmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* a, int lda, float* x, int incx)
{
    aclblasStatus_t valRet = StbmvValidateParams(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    if (valRet != ACLBLAS_STATUS_SUCCESS) return valRet;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    const int absIncx = std::abs(incx);
    const size_t xLen = static_cast<size_t>(absIncx) * (n - 1) + 1;
    std::vector<float> xCopy(x, x + xLen);

    const bool transIsN = (trans == ACLBLAS_OP_N);
    for (int row = 0; row < n; ++row) {
        int colStart = 0;
        int colEnd = 0;
        if (transIsN) {
            if (uplo == ACLBLAS_UPPER) {
                colStart = row;
                colEnd = std::min(n, row + k + 1);
            } else {
                colStart = std::max(0, row - k);
                colEnd = row + 1;
            }
        } else {
            if (uplo == ACLBLAS_UPPER) {
                colStart = std::max(0, row - k);
                colEnd = row + 1;
            } else {
                colStart = row;
                colEnd = std::min(n, row + k + 1);
            }
        }

        float acc = 0.0f;
        for (int col = colStart; col < colEnd; ++col) {
            float aVal = transIsN ? StbmvGetBandValue(a, uplo, diag, row, col, n, k, lda)
                                  : StbmvGetBandValue(a, uplo, diag, col, row, n, k, lda);
            acc += aVal * xCopy[StbmvVectorIndex(col, n, incx)];
        }
        x[StbmvVectorIndex(row, n, incx)] = acc;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

