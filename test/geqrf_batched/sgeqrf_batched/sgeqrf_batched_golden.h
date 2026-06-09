/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEQRF_BATCHED_GOLDEN_H
#define SGEQRF_BATCHED_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// CPU reference implementation for single-batch sgeqrf (Householder QR factorization).
// All intermediate calculations use double precision to minimize floating-point error
// accumulation across sequential Householder reflections. Input/output remain float32.
// Follows the standard LAPACK DLARFG convention for Householder reflector generation.
//
// On exit:
//   - Upper triangle (including diagonal) of A contains R.
//   - Below diagonal of A contains the Householder vectors v (v[0]=1 implicit).
//   - tau[k] contains the scalar factor for the k-th reflector.
//
// Algorithm per column i:
//   1. sigma = ||A[i+1:m-1, i]||^2
//   2. If sigma == 0: tau=0, no update needed.
//      Else: norm = sqrt(x1^2 + sigma), alpha = -sign(x1)*norm,
//            tau = (alpha - x1) / alpha, v = A[i+1:m, i] / (x1 - alpha)
//   3. Trailing submatrix update: A' = (I - tau * v * v^T) * A
inline void sgeqrf_single_batch(int m, int n, float* A, int lda, float* tau)
{
    int k_max = std::min(m, n);
    for (int i = 0; i < k_max; i++) {
        // Step 1: sigma = ||A[i+1:m-1, i]||^2  (accumulated in double)
        double sigma = 0.0;
        for (int row = i + 1; row < m; row++) {
            double val = static_cast<double>(A[row + i * lda]);
            sigma += val * val;
        }

        // Step 2: compute tau and alpha (in double)
        double x1 = static_cast<double>(A[i + i * lda]);
        double tauVal = 0.0;
        double alpha = 0.0;

        if (sigma == 0.0) {
            // LAPACK DLARFG convention: if x1 < 0, set tau=2 and alpha=-x1
            // so that R[i,i] = alpha is positive
            if (x1 < 0.0) {
                tauVal = 2.0;
                alpha = -x1;
            } else {
                tauVal = 0.0;
                alpha = x1;
            }
        } else {
            double normX = std::sqrt(sigma + x1 * x1);
            alpha = (x1 >= 0.0) ? -normX : normX;
            tauVal = (alpha - x1) / alpha;
        }

        // tau == 0 fast path: skip normalization and rank-1 update
        if (tauVal == 0.0) {
            A[i + i * lda] = static_cast<float>(alpha);
            tau[i] = static_cast<float>(tauVal);
            continue;
        }

        // Step 3: normalize v, write back A column (compute in double, store as float)
        //   v[j] = A[j, i] / (x1 - alpha) for j in [i+1, m)
        //   v[i] = 1 (implicit, not stored)
        double vScale = x1 - alpha;
        for (int row = i + 1; row < m; row++) {
            A[row + i * lda] = static_cast<float>(static_cast<double>(A[row + i * lda]) / vScale);
        }
        A[i + i * lda] = static_cast<float>(alpha);
        tau[i] = static_cast<float>(tauVal);

        // Step 4: rank-1 update of trailing submatrix (accumulated in double)
        //   For each column c in [i+1, n):
        //     dot = v^T * A[:, c] = A[i,c] + sum(v[j]*A[j,c], j=i+1..m-1)
        //     A[:, c] -= tau * dot * v
        for (int c = i + 1; c < n; c++) {
            // Compute dot = v^T * A[:, c]  (accumulated in double)
            // v[i] = 1 (implicit), so start with A[i, c]
            double dot = static_cast<double>(A[i + c * lda]);
            for (int row = i + 1; row < m; row++) {
                dot += static_cast<double>(A[row + i * lda]) * static_cast<double>(A[row + c * lda]);
            }

            double coeff = tauVal * dot;
            // Update row i (v[i] = 1)
            A[i + c * lda] = static_cast<float>(static_cast<double>(A[i + c * lda]) - coeff);
            // Update rows i+1..m-1
            for (int row = i + 1; row < m; row++) {
                A[row + c * lda] = static_cast<float>(
                    static_cast<double>(A[row + c * lda]) - coeff * static_cast<double>(A[row + i * lda]));
            }
        }
    }
}

// Batched golden: runs sgeqrf independently on each batch.
// Aarray[j] points to a column-major m×n matrix with leading dimension lda.
// TauArray[j] points to a vector of min(m,n) floats.
// info[j] is set to 0 on success.
inline aclblasStatus_t aclblasSgeqrfBatched_cpu(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0) {
        if (info != nullptr)
            info[0] = -1;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        if (info != nullptr)
            info[0] = -2;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        if (info != nullptr)
            info[0] = -4;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        if (info != nullptr)
            info[0] = -7;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize == 0 || m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (Aarray == nullptr || TauArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr && TauArray[b] != nullptr) {
            sgeqrf_single_batch(m, n, Aarray[b], lda, TauArray[b]);
        }
        if (info != nullptr)
            info[b] = 0;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SGEQRF_BATCHED_GOLDEN_H
