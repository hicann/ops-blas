#ifndef SSYRK_GOLDEN_H
#define SSYRK_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

static const int SSYRK_CPU_VALID_OK = 0x7FFFFFFF;

static bool IsValidUplo(aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER;
}

static bool IsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

static int ValidateSsyrkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, int lda, int ldc,
    const float* alpha, const float* beta, const float* A, const float* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED);
    }
    if (!IsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (!IsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
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
    return SSYRK_CPU_VALID_OK;
}

static void ClearMatrix(float* C, int n, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            C[static_cast<size_t>(j) * ldc + i] = 0.0f;
        }
    }
}

static void ScaleMatrix(float* C, int n, int ldc, float betaVal)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            size_t idx = static_cast<size_t>(j) * ldc + i;
            C[idx] = betaVal * C[idx];
        }
    }
}

inline aclblasStatus_t aclblasSsyrk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const float* A, int lda,
    const float* beta, float* C, int ldc)
{
    int st = ValidateSsyrkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != SSYRK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    if (alphaVal == 0.0f && betaVal == 0.0f) {
        ClearMatrix(C, n, ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (k == 0) {
        if (betaVal == 0.0f) {
            ClearMatrix(C, n, ldc);
        } else {
            ScaleMatrix(C, n, ldc, betaVal);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ssyrk(
        CblasColMajor,
        ToCblasUplo(uplo),
        ToCblasOp(trans),
        n, k,
        alphaVal, A, lda,
        betaVal, C, ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SSYRK_GOLDEN_H
