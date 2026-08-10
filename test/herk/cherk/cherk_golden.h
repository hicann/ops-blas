/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHERK_GOLDEN_H
#define CHERK_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

// Sentinel: all validations passed, proceed to compute.
static const int CHERK_CPU_VALID_OK = 0x7FFFFFFF;

static bool CherkIsValidUplo(aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER;
}

static bool CherkIsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

// Parameter validation aligned with ssyrk_golden.h; complex version differs
// only in A/C pointer types (aclblasComplex*). See 测试方案 §2.4.
static int ValidateCherkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, int lda, int ldc,
    const float* alpha, const float* beta, const aclblasComplex* A, const aclblasComplex* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!CherkIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (!CherkIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
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
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CHERK_CPU_VALID_OK;
}

// CPU golden for aclblasCherk. Signature mirrors the NPU API exactly so the
// test harness can swap NPU/CPU calls one-for-one. Returns aclblasStatus_t and
// applies the same validation as the NPU operator (CBLAS would crash on null
// pointers, so validation must precede the cblas_cherk call).
//
// Key design point (测试方案 §2.6): BLAS standard CHERK only accepts
// CblasNoTrans / CblasConjTrans — CblasTrans is illegal because a pure
// transpose would break Hermitian symmetry. The 需求文档 mandates that both
// trans='T' and trans='C' map to the conjugate-transpose A^H. Therefore:
//   trans == OP_N  -> CblasNoTrans
//   trans == OP_T  -> CblasConjTrans   (NOT CblasTrans)
//   trans == OP_C  -> CblasConjTrans
inline aclblasStatus_t aclblasCherk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc)
{
    int st = ValidateCherkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != CHERK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    // §2.6: T/C both map to CblasConjTrans for cblas_cherk.
    CBLAS_TRANSPOSE cblasTrans = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasConjTrans;

    // aclblasComplex = {float real; float imag} is binary-compatible with
    // OpenBLAS's complex layout, so the void* cast is a pure reinterpret.
    cblas_cherk(
        CblasColMajor,
        ToCblasUplo(uplo),
        cblasTrans,
        n, k,
        alphaVal,
        static_cast<const void*>(A), lda,
        betaVal,
        static_cast<void*>(C), ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CHERK_GOLDEN_H
