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

#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t ValidateCher2kCpuEnums(aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (trans == ACLBLAS_OP_T)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INVALID_ENUM;
}

inline aclblasStatus_t ValidateCher2kCpuShape(aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    if (n < 0 || k < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const int minLd = trans == ACLBLAS_OP_N ? std::max(1, n) : std::max(1, k);
    return lda >= minLd && ldb >= minLd && ldc >= std::max(1, n) ? ACLBLAS_STATUS_SUCCESS :
                                                                   ACLBLAS_STATUS_INVALID_VALUE;
}

inline aclblasStatus_t ValidateCher2kCpuPointers(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* b, const float* beta,
    aclblasComplex* c)
{
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const bool noProduct = k == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f);
    if (n > 0 && c == nullptr && !(noProduct && *beta == 0.0f))
        return ACLBLAS_STATUS_INVALID_VALUE;
    return n > 0 && k > 0 && (a == nullptr || b == nullptr) ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateCher2kCpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    aclblasStatus_t status = ValidateCher2kCpuEnums(uplo, trans);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ValidateCher2kCpuShape(trans, n, k, lda, ldb, ldc);
    return status == ACLBLAS_STATUS_SUCCESS ? ValidateCher2kCpuPointers(n, k, alpha, a, b, beta, c) : status;
}

inline void Cher2kScaleTriangle(aclblasFillMode_t uplo, int n, float beta, aclblasComplex* c, int ldc)
{
    for (int col = 0; col < n; ++col) {
        const int rowBegin = uplo == ACLBLAS_UPPER ? 0 : col;
        const int rowEnd = uplo == ACLBLAS_UPPER ? col : n - 1;
        for (int row = rowBegin; row <= rowEnd; ++row) {
            const size_t index = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            c[index].real = beta == 0.0f ? 0.0f : beta * c[index].real;
            c[index].imag = row == col ? 0.0f : (beta == 0.0f ? 0.0f : beta * c[index].imag);
        }
    }
}

inline aclblasStatus_t aclblasCher2k_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc)
{
    const aclblasStatus_t status = ValidateCher2kCpu(handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0)
        return status;
    if (k == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f)) {
        if (*beta == 1.0f || c == nullptr)
            return ACLBLAS_STATUS_SUCCESS;
        Cher2kScaleTriangle(uplo, n, *beta, c, ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }
    cblas_cher2k(
        CblasColMajor, ToCblasUplo(uplo), trans == ACLBLAS_OP_N ? CblasNoTrans : CblasConjTrans, n, k,
        static_cast<const void*>(alpha), static_cast<const void*>(a), lda, static_cast<const void*>(b), ldb, *beta,
        static_cast<void*>(c), ldc);
    return ACLBLAS_STATUS_SUCCESS;
}
