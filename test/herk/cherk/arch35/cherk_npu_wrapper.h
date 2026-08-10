/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHERK_NPU_WRAPPER_H
#define CHERK_NPU_WRAPPER_H

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CherkDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~CherkDeviceBuffers()
    {
        if (dAlpha) aclrtFree(dAlpha);
        if (dA) aclrtFree(dA);
        if (dBeta) aclrtFree(dBeta);
        if (dC) aclrtFree(dC);
    }
};

static aclblasStatus_t CherkAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate+copy only when hostPtr is non-null and bytes > 0; leaves devPtr
// null otherwise so the caller can pass nullptr straight through to the API
// (this is how the null_alpha / nullA / nullC cases reach the operator).
static aclblasStatus_t CherkTryAllocAndCopy(
    const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CherkAllocAndCopy(devPtr, hostPtr, bytes);
}

// NPU wrapper for aclblasCherk. Mirrors aclblasSsyrk_npu with complex-specific
// element sizes (aclblasComplex = 8 bytes). When handle is null or n<=0 it
// forwards directly so the operator's own validation paths are exercised.
//
// nullAlpha handling (§8 遗留问题 3): when nullAlpha=true the wrapper passes
// nullptr for alpha (no device alloc/copy), exercising the API's INVALID_VALUE
// path. Same pattern applies to nullA / nullC.
inline aclblasStatus_t aclblasCherk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc, bool nullAlpha = false)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCherk(handle, uplo, trans, n, k,
            (nullAlpha ? nullptr : alpha), A, lda, beta, C, ldc);
    }

    const int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    constexpr size_t scalarBytes = sizeof(float);

    // When nullAlpha is set, force the alpha pointer seen by the API to null.
    const float* effAlpha = nullAlpha ? nullptr : alpha;

    CherkDeviceBuffers bufs;
    aclblasStatus_t st;

    st = CherkTryAllocAndCopy(effAlpha, scalarBytes, bufs.dAlpha);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = CherkTryAllocAndCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = CherkTryAllocAndCopy(beta, scalarBytes, bufs.dBeta);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = CherkTryAllocAndCopy(C, cBytes, bufs.dC);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasCherk(
        handle, uplo, trans, n, k,
        bufs.dAlpha ? static_cast<const float*>(bufs.dAlpha) : effAlpha,
        bufs.dA ? static_cast<const aclblasComplex*>(bufs.dA) : A,
        lda,
        bufs.dBeta ? static_cast<const float*>(bufs.dBeta) : beta,
        bufs.dC ? static_cast<aclblasComplex*>(bufs.dC) : C,
        ldc);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (C != nullptr && bufs.dC != nullptr) {
        aclRet = aclrtMemcpy(C, cBytes, bufs.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    return ret;
}

#endif // CHERK_NPU_WRAPPER_H
