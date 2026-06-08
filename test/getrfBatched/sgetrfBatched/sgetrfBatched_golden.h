/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGETRF_BATCHED_GOLDEN_H
#define SGETRF_BATCHED_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

/**
 * CPU golden reference for batched LU factorization (sgetrf).
 *
 * Signature matches the BLAS API exactly.
 * Operates on host-side pointer array (each Aarray[i] points to host memory).
 *
 * Storage: column-major, A(row, col) = A[col * lda + row]
 * Pivot:   1-indexed (LAPACK convention), PivotArray[batch * n + k]
 * Info:    infoArray[batch] = 0 if success, k (1-indexed) if U(k-1,k-1) == 0
 */
inline aclblasStatus_t ValidateGetrfParams(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0 || batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (Aarray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (PivotArray != nullptr && infoArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline void FactorizeSingleMatrix(float* A, int n, int lda, int* piv, bool usePivot, int& info)
{
    info = 0;
    for (int k = 0; k < n; k++) {
        int pivotRow = k;
        if (usePivot) {
            float maxVal = std::abs(A[k + k * lda]);
            for (int i = k + 1; i < n; i++) {
                float absVal = std::abs(A[i + k * lda]);
                if (absVal > maxVal) {
                    maxVal = absVal;
                    pivotRow = i;
                }
            }
            piv[k] = pivotRow + 1;
        }

        float diagVal = A[pivotRow + k * lda];
        if (diagVal == 0.0f) {
            info = k + 1;
            break;
        }

        if (pivotRow != k) {
            for (int j = 0; j < n; j++) {
                std::swap(A[k + j * lda], A[pivotRow + j * lda]);
            }
        }

        diagVal = A[k + k * lda];
        for (int i = k + 1; i < n; i++) {
            A[i + k * lda] /= diagVal;
        }

        for (int j = k + 1; j < n; j++) {
            float uKJ = A[k + j * lda];
            for (int i = k + 1; i < n; i++) {
                A[i + j * lda] -= A[i + k * lda] * uKJ;
            }
        }
    }
}

inline aclblasStatus_t aclblasSgetrfBatched_cpu(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    aclblasStatus_t validRet = ValidateGetrfParams(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool usePivot = (PivotArray != nullptr);
    for (int b = 0; b < batchSize; b++) {
        int* piv = usePivot ? (PivotArray + b * n) : nullptr;
        int info = 0;
        FactorizeSingleMatrix(Aarray[b], n, lda, piv, usePivot, info);
        if (infoArray != nullptr) {
            infoArray[b] = info;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SGETRF_BATCHED_GOLDEN_H
