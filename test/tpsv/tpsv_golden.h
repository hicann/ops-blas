/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TPSV_CPU_H
#define TPSV_CPU_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline size_t TpsvPackedUpperIdxCpu(int i, int j)
{
    return static_cast<size_t>(i + j * (j + 1) / 2);
}

inline size_t TpsvPackedLowerIdxCpu(int i, int j, int n)
{
    return static_cast<size_t>(i + (2 * n - j - 1) * j / 2);
}

inline float TpsvGetDiagCpu(const std::vector<float>& ap, int row, aclblasFillMode_t uplo, int n)
{
    if (uplo == ACLBLAS_LOWER) {
        return ap[TpsvPackedLowerIdxCpu(row, row, n)];
    }
    return ap[TpsvPackedUpperIdxCpu(row, row)];
}

inline float TpsvGetElemOffDiagCpu(
    const std::vector<float>& ap, int row, int col, aclblasFillMode_t uplo, aclblasOperation_t trans, int n)
{
    if (uplo == ACLBLAS_LOWER) {
        if (trans == ACLBLAS_OP_N) {
            return ap[TpsvPackedLowerIdxCpu(row, col, n)];
        }
        return ap[TpsvPackedLowerIdxCpu(col, row, n)];
    }
    if (trans == ACLBLAS_OP_N) {
        return ap[TpsvPackedUpperIdxCpu(row, col)];
    }
    return ap[TpsvPackedUpperIdxCpu(col, row)];
}

// reference implementation — same signature as aclblasStpsv
inline aclblasStatus_t aclblasStpsv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* ap, float* x, int incx)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ap == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
    std::vector<float> apCopy(ap, ap + apLen);

    int absIncx = std::abs(incx);
    size_t xLen = static_cast<size_t>(absIncx * (n - 1) + 1);
    std::vector<float> xCopy(x, x + xLen);

    bool forward =
        (uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_N) || (uplo == ACLBLAS_UPPER && trans != ACLBLAS_OP_N);

    if (forward) {
        for (int i = 0; i < n; ++i) {
            int xiIdx = (incx >= 0) ? (i * incx) : ((n - 1 - i) * absIncx);
            float sum = xCopy[xiIdx];
            for (int j = 0; j < i; ++j) {
                sum -= TpsvGetElemOffDiagCpu(apCopy, i, j, uplo, trans, n) *
                       xCopy[(incx >= 0) ? (j * incx) : ((n - 1 - j) * absIncx)];
            }
            if (diag == ACLBLAS_NON_UNIT) {
                sum /= TpsvGetDiagCpu(apCopy, i, uplo, n);
            }
            xCopy[xiIdx] = sum;
        }
    } else {
        for (int i = n - 1; i >= 0; --i) {
            int xiIdx = (incx >= 0) ? (i * incx) : ((n - 1 - i) * absIncx);
            float sum = xCopy[xiIdx];
            for (int j = i + 1; j < n; ++j) {
                sum -= TpsvGetElemOffDiagCpu(apCopy, i, j, uplo, trans, n) *
                       xCopy[(incx >= 0) ? (j * incx) : ((n - 1 - j) * absIncx)];
            }
            if (diag == ACLBLAS_NON_UNIT) {
                sum /= TpsvGetDiagCpu(apCopy, i, uplo, n);
            }
            xCopy[xiIdx] = sum;
        }
    }

    std::copy(xCopy.begin(), xCopy.end(), x);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // TPSV_CPU_H
