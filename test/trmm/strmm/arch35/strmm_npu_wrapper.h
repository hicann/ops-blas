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

inline aclblasStatus_t aclblasStrmm_npu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t transA,
    aclblasDiagType_t diag,
    int64_t m,
    int64_t n,
    const float* alpha,
    const float* A,
    int64_t lda,
    float* B,
    int64_t ldb)
{
    if (handle == nullptr || m <= 0 || n <= 0 ||
        alpha == nullptr || A == nullptr || B == nullptr) {
        return aclblasStrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(m) * static_cast<size_t>(ldb) * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;

    aclError aclRet = AllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = AllocCopyH2D(dB, B, bBytes);
    if (aclRet != ACL_SUCCESS) { FreeDev(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }

    aclblasStatus_t ret = aclblasStrmm(
        handle, side, uplo, transA, diag, m, n,
        alpha,
        static_cast<const float*>(dA), lda,
        static_cast<float*>(dB), ldb);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasGetStream(handle, &stream);
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        aclError d2hRet = aclrtMemcpy(B, bBytes, dB, bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (d2hRet != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    FreeDev(dA);
    FreeDev(dB);
    return ret;
}

#endif // STRMM_NPU_H
