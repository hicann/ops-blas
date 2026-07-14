/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_NPU_H
#define STRMM_NPU_H

#include "acl/acl.h"
#include "cann_ops_blas.h"

static inline aclError AllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(dPtr); dPtr = nullptr; }
    return ret;
}

static inline void FreeDev(void* dPtr)
{
    if (dPtr) aclrtFree(dPtr);
}

static inline aclblasStatus_t StrmmCopyB2C(void* dC, const void* dB,
    int m, int n, int ldb, int ldc, size_t cBytes, size_t bBytes)
{
    if (ldb == ldc) {
        if (aclrtMemcpy(dC, cBytes, dB, bBytes, ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMemset(dC, cBytes, 0, cBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    size_t rowBytes = static_cast<size_t>(n) * sizeof(float);
    for (int i = 0; i < m; ++i) {
        size_t offset = static_cast<size_t>(i) * static_cast<size_t>(ldc) * sizeof(float);
        if (aclrtMemcpy(static_cast<float*>(dC) + static_cast<size_t>(i) * ldc,
                cBytes - offset,
                static_cast<const float*>(dB) + static_cast<size_t>(i) * ldb,
                rowBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t StrmmSyncAndCopyD2H(aclblasHandle handle,
    const void* dC, float* C, size_t cBytes)
{
    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);
    if (stream != nullptr) {
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrmm_npu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc)
{
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (handle == nullptr || m <= 0 || n <= 0 ||
        alpha == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) ||
        lda < aDim || ldb < n || ldc < n) {
        return aclblasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    }

    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(m) * static_cast<size_t>(ldb) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(m) * static_cast<size_t>(ldc) * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    aclError aclRet = AllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = AllocCopyH2D(dB, B, bBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { FreeDev(dA); FreeDev(dB); return ACLBLAS_STATUS_ALLOC_FAILED; }

    aclblasStatus_t cRet = StrmmCopyB2C(dC, dB, m, n, ldb, ldc, cBytes, bBytes);
    if (cRet != ACLBLAS_STATUS_SUCCESS) {
        FreeDev(dA); FreeDev(dB); FreeDev(dC);
        return cRet;
    }

    aclblasStatus_t ret = aclblasStrmm(
        handle, side, uplo, trans, diag, m, n,
        alpha,
        static_cast<const float*>(dA), lda,
        static_cast<const float*>(dB), ldb,
        static_cast<float*>(dC), ldc);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = StrmmSyncAndCopyD2H(handle, dC, C, cBytes);
    }

    FreeDev(dA);
    FreeDev(dB);
    FreeDev(dC);
    return ret;
}

#endif // STRMM_NPU_H
