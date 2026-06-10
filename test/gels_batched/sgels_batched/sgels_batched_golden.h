/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GELS_BATCHED_GOLDEN_H
#define GELS_BATCHED_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace gels_golden {

inline void applyHouseholderLeft(int m, int n, int k, const float* v, float tau, float* C, int ldc)
{
    if (tau == 0.0f)
        return;
    for (int j = 0; j < n; j++) {
        float dot = C[k + j * ldc];
        for (int i = k + 1; i < m; i++) {
            dot += v[i] * C[i + j * ldc];
        }
        C[k + j * ldc] -= tau * dot;
        for (int i = k + 1; i < m; i++) {
            C[i + j * ldc] -= tau * dot * v[i];
        }
    }
}

inline void applyHouseholderRight(int m, int n, int k, const float* v, float tau, float* C, int ldc)
{
    if (tau == 0.0f)
        return;
    for (int i = 0; i < m; i++) {
        float dot = C[i + k * ldc];
        for (int j = k + 1; j < n; j++) {
            dot += v[j] * C[i + j * ldc];
        }
        C[i + k * ldc] -= tau * dot;
        for (int j = k + 1; j < n; j++) {
            C[i + j * ldc] -= tau * dot * v[j];
        }
    }
}

inline float generateHouseholder(int n, const float* x, float* v)
{
    if (n <= 0)
        return 0.0f;

    v[0] = 1.0f;
    for (int i = 1; i < n; i++)
        v[i] = x[i];

    float xnorm = 0.0f;
    for (int i = 1; i < n; i++)
        xnorm += x[i] * x[i];

    if (xnorm == 0.0f && x[0] >= 0.0f)
        return 0.0f;
    if (xnorm == 0.0f && x[0] < 0.0f)
        return 2.0f;

    float x0 = x[0];
    float beta = -std::copysign(1.0f, x0) * std::sqrt(x0 * x0 + xnorm);
    float diff = x0 - beta;

    if (std::abs(diff) > 1e-30f) {
        float scale = 1.0f / diff;
        for (int i = 1; i < n; i++)
            v[i] *= scale;
    }

    return (beta - x0) / beta;
}

// Unified triangular solve: upper=true for back-substitution, upper=false for forward-substitution
inline void solveTriangular(int n, int nrhs, const float* A, int lda, float* C, int ldc, bool upper)
{
    for (int j = 0; j < nrhs; j++) {
        for (int ii = 0; ii < n; ii++) {
            int i = upper ? (n - 1 - ii) : ii;
            float sum = C[i + j * ldc];
            int kStart = upper ? (i + 1) : 0;
            int kEnd = upper ? n : i;
            for (int k = kStart; k < kEnd; k++) {
                sum -= A[i + k * lda] * C[k + j * ldc];
            }
            if (std::abs(A[i + i * lda]) > 1e-30f) {
                C[i + j * ldc] = sum / A[i + i * lda];
            } else {
                C[i + j * ldc] = 0.0f;
            }
        }
    }
}

inline float computeBeta(const float* col, const float* hv, int len, float tau)
{
    if (tau == 0.0f)
        return col[0];
    float vdotx = col[0];
    for (int i = 1; i < len; i++)
        vdotx += hv[i] * col[i];
    return col[0] - tau * vdotx;
}

// QR factorization + solve for overdetermined systems (m >= n)
inline int solveOverdetermined(int m, int n, int nrhs, float* A, int lda, float* C, int ldc)
{
    for (int k = 0; k < n; k++) {
        std::vector<float> col(m - k);
        for (int i = 0; i < m - k; i++)
            col[i] = A[(k + i) + k * lda];

        std::vector<float> hv(m - k);
        float tau = generateHouseholder(m - k, col.data(), hv.data());

        if (k + 1 < n)
            applyHouseholderLeft(m - k, n - k - 1, 0, hv.data(), tau, &A[k + (k + 1) * lda], lda);
        applyHouseholderLeft(m - k, nrhs, 0, hv.data(), tau, &C[k], ldc);
        A[k + k * lda] = computeBeta(col.data(), hv.data(), m - k, tau);
    }

    for (int k = 0; k < n; k++) {
        if (std::abs(A[k + k * lda]) < 1e-10f)
            return k + 1;
    }
    solveTriangular(n, nrhs, A, lda, C, ldc, true);
    return 0;
}

// LQ factorization + solve for underdetermined systems (m < n)
inline int solveUnderdetermined(int m, int n, int nrhs, float* A, int lda, float* C, int ldc)
{
    std::vector<float> taus(m);
    for (int k = 0; k < m; k++) {
        std::vector<float> row(n - k);
        for (int j = 0; j < n - k; j++)
            row[j] = A[k + (k + j) * lda];

        std::vector<float> hv(n - k);
        taus[k] = generateHouseholder(n - k, row.data(), hv.data());

        if (k + 1 < m)
            applyHouseholderRight(m - k - 1, n - k, 0, hv.data(), taus[k], &A[(k + 1) + k * lda], lda);

        A[k + k * lda] = computeBeta(row.data(), hv.data(), n - k, taus[k]);
        for (int j = 1; j < n - k; j++)
            A[k + (k + j) * lda] = hv[j];
    }

    for (int k = 0; k < m; k++) {
        if (std::abs(A[k + k * lda]) < 1e-10f)
            return k + 1;
    }

    solveTriangular(m, nrhs, A, lda, C, ldc, false);

    for (int j = 0; j < nrhs; j++)
        for (int i = m; i < n; i++)
            C[i + j * ldc] = 0.0f;

    for (int k = 0; k < m; k++) {
        std::vector<float> hv(n - k);
        hv[0] = 1.0f;
        for (int j = 1; j < n - k; j++)
            hv[j] = A[k + (k + j) * lda];
        applyHouseholderLeft(n - k, nrhs, 0, hv.data(), taus[k], &C[k], ldc);
    }
    return 0;
}

// OP_T: transpose A, then solve with transposed matrix
inline int solveTransposed(int m, int n, int nrhs, const float* A, int lda, float* C, int ldc)
{
    std::vector<float> AT(static_cast<size_t>(n) * m);
    int atlda = n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j + i * atlda] = A[i + j * lda];

    if (n >= m)
        return solveOverdetermined(n, m, nrhs, AT.data(), atlda, C, ldc);
    else
        return solveUnderdetermined(n, m, nrhs, AT.data(), atlda, C, ldc);
}

} // namespace gels_golden

static inline aclblasStatus_t validateGelsCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0 || nrhs < 0 || batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max({1, m, n}))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchSize > 0 && (Aarray == nullptr || Carray == nullptr))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (devInfo == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

static inline int solveBatch(
    aclblasOperation_t trans, int m, int n, int nrhs, float* Awork, int lda, float* Cwork, int ldc)
{
    if (trans == ACLBLAS_OP_T)
        return gels_golden::solveTransposed(m, n, nrhs, Awork, lda, Cwork, ldc);
    if (m >= n)
        return gels_golden::solveOverdetermined(m, n, nrhs, Awork, lda, Cwork, ldc);
    return gels_golden::solveUnderdetermined(m, n, nrhs, Awork, lda, Cwork, ldc);
}

inline aclblasStatus_t aclblasSgelsBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize)
{
    aclblasStatus_t validRet =
        validateGelsCpuParams(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, devInfo, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;

    if (m == 0 || n == 0 || nrhs == 0 || batchSize == 0) {
        *devInfo = 0;
        return ACLBLAS_STATUS_SUCCESS;
    }

    int globalInfo = 0;
    const int solRows = (trans == ACLBLAS_OP_N) ? n : m;

    for (int b = 0; b < batchSize; b++) {
        std::vector<float> Awork(static_cast<size_t>(lda) * n);
        std::vector<float> Cwork(static_cast<size_t>(ldc) * nrhs, 0.0f);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Awork[i + j * lda] = Aarray[b][i + j * lda];

        int rhsRows = (trans == ACLBLAS_OP_N) ? m : n;
        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < rhsRows; i++)
                Cwork[i + j * ldc] = Carray[b][i + j * ldc];

        int info = solveBatch(trans, m, n, nrhs, Awork.data(), lda, Cwork.data(), ldc);

        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < solRows; i++)
                Carray[b][i + j * ldc] = Cwork[i + j * ldc];

        if (info != 0 && globalInfo == 0)
            globalInfo = info;
    }

    *devInfo = globalInfo;
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GELS_BATCHED_GOLDEN_H
