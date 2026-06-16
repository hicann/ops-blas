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

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Read a value from symmetric matrix A (row-major, only one triangle stored).
// The missing triangle is reconstructed by transposing indices.
static inline float SsymmGetSymValue(const float* a, int64_t lda,
    aclblasFillMode_t uplo, int64_t row, int64_t col)
{
    if (uplo == ACLBLAS_LOWER) {
        return (row >= col) ? a[row * lda + col] : a[col * lda + row];
    }
    return (row <= col) ? a[row * lda + col] : a[col * lda + row];
}

// CPU reference implementation — signature matches aclblasSsymm exactly.
// Reproduces ValidateSsymmArgs parameter validation followed by the
// three-loop matrix multiply.
inline aclblasStatus_t aclblasSsymm_cpu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m,
    int64_t n,
    const float* alpha,
    const float* A,
    int64_t lda,
    const float* B,
    int64_t ldb,
    const float* beta,
    float* C,
    int64_t ldc)
{
    // --- parameter validation (mirrors ValidateSsymmArgs) ---
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0) return ACLBLAS_STATUS_INVALID_VALUE;

    // quick return
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    const int64_t aDim   = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const int64_t maxU32 = static_cast<int64_t>(UINT32_MAX);
    if (m > maxU32 || n > maxU32 || aDim > maxU32 ||
        lda > maxU32 || ldb > maxU32 || ldc > maxU32)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < aDim || ldb < n || ldc < n)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // --- computation ---
    // scale C by beta first
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            C[i * ldc + j] *= (*beta);
        }
    }

    // accumulate alpha * A_sym * B  (LEFT) or alpha * B * A_sym  (RIGHT)
    if (side == ACLBLAS_SIDE_LEFT) {
        // C[i][j] += alpha * sum_k( A_sym[i][k] * B[k][j] )
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int64_t k = 0; k < m; ++k) {
                    acc += static_cast<double>(SsymmGetSymValue(A, lda, uplo, i, k))
                         * static_cast<double>(B[k * ldb + j]);
                }
                C[i * ldc + j] += static_cast<float>((*alpha) * static_cast<float>(acc));
            }
        }
    } else {
        // C[i][j] += alpha * sum_k( B[i][k] * A_sym[k][j] )
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    acc += static_cast<double>(B[i * ldb + k])
                         * static_cast<double>(SsymmGetSymValue(A, lda, uplo, k, j));
                }
                C[i * ldc + j] += static_cast<float>((*alpha) * static_cast<float>(acc));
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

