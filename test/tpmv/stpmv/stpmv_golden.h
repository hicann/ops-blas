/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STPMV_GOLDEN_H
#define STPMV_GOLDEN_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Standard BLAS column-major packed lower triangular: AP(i,j) = AP[i + (2n-j-1)*j/2], 0 <= j <= i
static inline size_t PackedLowerRowIndex(int n, int i, int j)
{
    int64_t n64 = n;
    int64_t i64 = i;
    int64_t j64 = j;
    return static_cast<size_t>(i64 + (2 * n64 - j64 - 1) * j64 / 2);
}

// Standard BLAS column-major packed upper triangular: AP(i,j) = AP[i + j*(j+1)/2], 0 <= i <= j
static inline size_t PackedUpperRowIndex(int n, int i, int j)
{
    (void)n;
    int64_t i64 = i;
    int64_t j64 = j;
    return static_cast<size_t>(i64 + j64 * (j64 + 1) / 2);
}

// CPU reference implementation — in-place: x = op(A) * x
// Uses internal temporary buffer to avoid read-write conflicts.
inline aclblasStatus_t aclblasStpmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (AP == nullptr || x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    const int absIncx = std::abs(incx);

    // BLAS standard: for incx < 0, logical element j is at physical position (n-1-j)*|incx|
    auto getX = [&](int j) -> float {
        int offset = (incx >= 0) ? (j * absIncx) : ((n - 1 - j) * absIncx);
        return x[offset];
    };

    // Use temporary buffer to avoid read-write conflicts during in-place update
    std::vector<float> result(static_cast<size_t>(n), 0.0f);

    if (trans == ACLBLAS_OP_N) {
        // result = A * x
        if (uplo == ACLBLAS_LOWER) {
            // Lower triangular: A(i,j) defined for j <= i
            for (int i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < i; ++j) {
                    size_t idx = PackedLowerRowIndex(n, i, j);
                    acc += AP[idx] * getX(j);
                }
                // Diagonal element
                float a_ii = (diag == ACLBLAS_UNIT) ? 1.0f : AP[PackedLowerRowIndex(n, i, i)];
                acc += a_ii * getX(i);
                result[i] = acc;
            }
        } else {
            // Upper triangular: A(i,j) defined for i <= j
            for (int i = 0; i < n; ++i) {
                float acc = 0.0f;
                // Diagonal element
                float a_ii = (diag == ACLBLAS_UNIT) ? 1.0f : AP[PackedUpperRowIndex(n, i, i)];
                acc += a_ii * getX(i);
                for (int j = i + 1; j < n; ++j) {
                    size_t idx = PackedUpperRowIndex(n, i, j);
                    acc += AP[idx] * getX(j);
                }
                result[i] = acc;
            }
        }
    } else {
        // result = A^T * x (trans == T or C, same for real)
        // Initialize result to zero first
        for (int i = 0; i < n; ++i) {
            result[i] = 0.0f;
        }

        if (uplo == ACLBLAS_LOWER) {
            // Lower triangular: A(j,i) defined for i <= j
            for (int j = 0; j < n; ++j) {
                float xj = getX(j);
                for (int i = 0; i < j; ++i) {
                    size_t idx = PackedLowerRowIndex(n, j, i);
                    result[i] += AP[idx] * xj;
                }
                // Diagonal element A(j,j)
                float a_jj = (diag == ACLBLAS_UNIT) ? 1.0f : AP[PackedLowerRowIndex(n, j, j)];
                result[j] += a_jj * xj;
            }
        } else {
            // Upper triangular: A(j,i) defined for j <= i
            for (int j = 0; j < n; ++j) {
                float xj = getX(j);
                // Diagonal element A(j,j)
                float a_jj = (diag == ACLBLAS_UNIT) ? 1.0f : AP[PackedUpperRowIndex(n, j, j)];
                result[j] += a_jj * xj;
                for (int i = j + 1; i < n; ++i) {
                    size_t idx = PackedUpperRowIndex(n, j, i);
                    result[i] += AP[idx] * xj;
                }
            }
        }
    }

    // Copy result back to x (in-place), respecting incx stride
    for (int i = 0; i < n; ++i) {
        int offset = (incx >= 0) ? (i * absIncx) : ((n - 1 - i) * absIncx);
        x[offset] = result[i];
    }

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // STPMV_GOLDEN_H
