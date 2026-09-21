/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYRK_GOLDEN_H
#define CSYRK_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "cblas_threaded.h"

// Sentinel: all validations passed, proceed to compute.
static const int CSYRK_CPU_VALID_OK = 0x7FFFFFFF;

static bool CsyrkIsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

// CSYRK forms A*A^T with a *plain* transpose, so the conjugate transpose is not
// a legal choice here -- unlike CHERK, where OP_T and OP_C both map to A^H.
static bool CsyrkIsValidTrans(aclblasOperation_t trans) { return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T; }

// Parameter validation aligned with cherk_golden.h. The order matters and matches
// the NPU implementation: enum legality and leading dimensions are checked before
// the n == 0 quick return, pointers after it.
static int ValidateCsyrkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc,
    const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* A, const aclblasComplex* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!CsyrkIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (!CsyrkIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    bool isTrans = (trans != ACLBLAS_OP_N);
    int minLda = isTrans ? std::max(1, k) : std::max(1, n);
    if (lda < minLda) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYRK_CPU_VALID_OK;
}

// CPU golden for aclblasCsyrk. Signature mirrors the NPU API exactly so the test
// harness can swap NPU/CPU calls one-for-one. Validation precedes the cblas call
// because CBLAS would crash on null pointers.
//
// Unlike CHERK, alpha and beta are complex and are passed by pointer to
// cblas_csyrk. aclblasComplex = {float real; float imag} is binary-compatible
// with OpenBLAS's complex layout, so every cast below is a pure reinterpret.
inline aclblasStatus_t aclblasCsyrk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    int st = ValidateCsyrkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != CSYRK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    CBLAS_TRANSPOSE cblasTrans = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasTrans;

    // CsyrkThreaded splits the output into column panels across threads; each
    // element still accumulates over the full K extent in the same order, so the
    // result is bit-identical to a single cblas_csyrk call. See cblas_threaded.h.
    blas_test::CsyrkThreaded(
        ToCblasUplo(uplo), cblasTrans, n, k, reinterpret_cast<const float*>(alpha),
        reinterpret_cast<const float*>(A), lda, reinterpret_cast<const float*>(beta),
        reinterpret_cast<float*>(C), ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSYRK_GOLDEN_H
