/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STPTTR_NPU_H
#define STPTTR_NPU_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// NPU wrapper — same signature as aclblasStpttr.
// nullptr inputs are passed through without device allocation (for error-path testing).
inline aclblasStatus_t aclblasStpttr_npu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* ap,
    float* a,
    int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStpttr(handle, uplo, n, ap, a, lda);
    }

    const int allocN = std::max(1, n);
    const int allocLda = std::max(allocN, lda);
    const size_t apBytes = static_cast<size_t>(allocN) * (allocN + 1) / 2 * sizeof(float);
    const size_t aBytes  = static_cast<size_t>(allocLda) * allocN * sizeof(float);

    void* dP = nullptr;
    void* dA = nullptr;
    aclError aclRet;

    if (ap != nullptr) {
        aclRet = aclrtMalloc(&dP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dP, apBytes, ap, apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dP); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (a != nullptr) {
        aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { if (dP) aclrtFree(dP); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(dA, aBytes, a, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { if (dP) aclrtFree(dP); aclrtFree(dA); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }

    aclblasStatus_t ret = aclblasStpttr(handle, uplo, n,
        static_cast<const float*>(dP), static_cast<float*>(dA), lda);

    aclrtSynchronizeDevice();
    if (a != nullptr) {
        aclrtMemcpy(a, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dP) aclrtFree(dP);
    if (dA) aclrtFree(dA);
    return ret;
}

#endif // STPTTR_NPU_H
