/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYMM_NPU_H
#define SSYMM_NPU_H

#include <cstdint>

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

inline aclblasStatus_t aclblasSsymm_npu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m,
    int64_t n,
    const float* alpha,
    const float* A,
    int64_t lda,
    const float* B,
    int64_t ldb,
    const float* beta,
    float* C,
    int64_t ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0 ||
        alpha == nullptr || A == nullptr || B == nullptr ||
        beta == nullptr || C == nullptr) {
        return aclblasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t alphaBytes = sizeof(float);
    const size_t betaBytes = sizeof(float);
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(m) * static_cast<size_t>(ldb) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(m) * static_cast<size_t>(ldc) * sizeof(float);

    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    aclError aclRet = AllocCopyH2D(dAlpha, alpha, alphaBytes);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = AllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dAlpha); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = AllocCopyH2D(dB, B, bBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dAlpha); FreeDev(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = AllocCopyH2D(dBeta, beta, betaBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dAlpha); FreeDev(dA); FreeDev(dB); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = AllocCopyH2D(dC, C, cBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dAlpha); FreeDev(dA); FreeDev(dB); FreeDev(dBeta); return ACLBLAS_STATUS_ALLOC_FAILED; }

    aclblasStatus_t ret = aclblasSsymm(
        handle, side, uplo, m, n,
        static_cast<const float*>(dAlpha),
        static_cast<const float*>(dA), lda,
        static_cast<const float*>(dB), ldb,
        static_cast<const float*>(dBeta),
        static_cast<float*>(dC), ldc);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasGetStream(handle, &stream);
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        aclError memcpyRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (memcpyRet != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    FreeDev(dAlpha);
    FreeDev(dA);
    FreeDev(dB);
    FreeDev(dBeta);
    FreeDev(dC);
    return ret;
}

#endif // SSYMM_NPU_H
