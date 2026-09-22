/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYMM_GOLDEN_H
#define CSYMM_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "cblas_threaded.h"

// Sentinel: all validations passed, proceed to compute.
static const int CSYMM_CPU_VALID_OK = 0x7FFFFFFF;

static bool CsymmIsValidSide(aclblasSideMode_t side) { return side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT; }

static bool CsymmIsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

// Parameter validation aligned with the NPU implementation, including the order:
// enum legality and leading dimensions come before the m == 0 || n == 0 quick
// return, pointers after it.
static int ValidateCsymmCpuParams(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, int lda, int ldb, int ldc,
    const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* A, const aclblasComplex* B,
    const aclblasComplex* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!CsymmIsValidSide(side)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (!CsymmIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (m < 0 || n < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // A is square of side m for side=LEFT and n for side=RIGHT.
    int ldaMin = (side == ACLBLAS_SIDE_LEFT) ? std::max(1, m) : std::max(1, n);
    if (lda < ldaMin) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldb < std::max(1, m) || ldc < std::max(1, m)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (m == 0 || n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYMM_CPU_VALID_OK;
}

// CPU golden for aclblasCsymm. Signature mirrors the NPU API exactly so the test
// harness can swap NPU/CPU calls one-for-one. Validation precedes the cblas call
// because CBLAS would crash on null pointers.
//
// Both alpha and beta are complex and passed by pointer, because CSYMM's output is
// a general matrix and nothing constrains the scalars. aclblasComplex = {float
// real; float imag} is binary-compatible with OpenBLAS's complex layout, so every
// cast below is a pure reinterpret.
inline aclblasStatus_t aclblasCsymm_cpu(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    int st = ValidateCsymmCpuParams(handle, side, uplo, m, n, lda, ldb, ldc, alpha, beta, A, B, C);
    if (st != CSYMM_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    CBLAS_SIDE cblasSide = (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;

    // CsymmThreaded splits the dense output along whichever extent leaves A whole
    // and drives one cblas_csymm per chunk from its own thread, which leaves every
    // element's accumulation order untouched. See cblas_threaded.h.
    blas_test::CsymmThreaded(
        cblasSide, ToCblasUplo(uplo), m, n, reinterpret_cast<const float*>(alpha),
        reinterpret_cast<const float*>(A), lda, reinterpret_cast<const float*>(B), ldb,
        reinterpret_cast<const float*>(beta), reinterpret_cast<float*>(C), ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSYMM_GOLDEN_H
