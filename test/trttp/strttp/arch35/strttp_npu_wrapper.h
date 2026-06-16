/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasStrttp_npu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* a,
    int lda,
    float* ap)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStrttp(handle, uplo, n, a, lda, ap);
    }

    const int allocN = std::max(1, n);
    const int allocLda = std::max(allocN, lda);
    const size_t aBytes  = static_cast<size_t>(allocLda) * allocN * sizeof(float);
    const size_t apBytes = static_cast<size_t>(allocN) * (allocN + 1) / 2 * sizeof(float);

    void* dA = nullptr;
    void* dP = nullptr;
    aclError aclRet;

    if (a != nullptr) {
        aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dA, aBytes, a, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dA); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (ap != nullptr) {
        aclRet = aclrtMalloc(&dP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { if (dA) aclrtFree(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(dP, apBytes, ap, apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { if (dA) aclrtFree(dA); aclrtFree(dP); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }

    aclblasStatus_t ret = aclblasStrttp(handle, uplo, n,
        static_cast<const float*>(dA), lda, static_cast<float*>(dP));

    aclrtSynchronizeDevice();
    if (ap != nullptr) {
        aclrtMemcpy(ap, apBytes, dP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dA) aclrtFree(dA);
    if (dP) aclrtFree(dP);
    return ret;
}

