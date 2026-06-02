/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRSV_CPU_H
#define TRSV_CPU_H

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t ValidateTrsvCpuParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    int64_t lda, int64_t incx, const float* A, const float* x)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(int64_t(1), n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (A == nullptr || x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrsv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    const float* A, int64_t lda, float* x, int64_t incx)
{
    aclblasStatus_t st = ValidateTrsvCpuParams(handle, uplo, trans, diag, n, lda, incx, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const int ni = static_cast<int>(n);
    const int ldai = static_cast<int>(lda);
    const int absIncx = std::abs(static_cast<int>(incx));
    const bool isUnit = (diag == ACLBLAS_UNIT);
    const bool isTranspose = (trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C);

    // Determine solve direction:
    //   forward (i = 0..n-1): LOWER+N, UPPER+T/C
    //   backward (i = n-1..0): UPPER+N, LOWER+T/C
    const bool isForward = (uplo == ACLBLAS_LOWER && !isTranspose) || (uplo == ACLBLAS_UPPER && isTranspose);

    // Helper: map logical index i into strided x buffer.
    auto xIdx = [&](int i) -> int { return (incx >= 0) ? (i * static_cast<int>(incx)) : ((ni - 1 - i) * absIncx); };

    if (isForward) {
        for (int i = 0; i < ni; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                // Forward: use columns j < i.
                // LOWER+N: A[i][j] = A[i + j*lda]
                // UPPER+T: A^T[i][j] = A[j][i] = A[j + i*lda]
                float aElem = isTranspose ? A[j + i * ldai] : A[i + j * ldai];
                sum += static_cast<double>(aElem) * static_cast<double>(x[xIdx(j)]);
            }
            float diagVal = isUnit ? 1.0f : A[i + i * ldai];
            x[xIdx(i)] = static_cast<float>((static_cast<double>(x[xIdx(i)]) - sum) / static_cast<double>(diagVal));
        }
    } else {
        for (int i = ni - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < ni; j++) {
                // Backward: use columns j > i.
                // UPPER+N: A[i][j] = A[i + j*lda]
                // LOWER+T: A^T[i][j] = A[j][i] = A[j + i*lda]
                float aElem = isTranspose ? A[j + i * ldai] : A[i + j * ldai];
                sum += static_cast<double>(aElem) * static_cast<double>(x[xIdx(j)]);
            }
            float diagVal = isUnit ? 1.0f : A[i + i * ldai];
            x[xIdx(i)] = static_cast<float>((static_cast<double>(x[xIdx(i)]) - sum) / static_cast<double>(diagVal));
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // TRSV_CPU_H
