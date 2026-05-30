/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSWAP_NPU_H
#define SSWAP_NPU_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// NPU wrapper — same signature as aclblasSswap.
// nullptr inputs are passed through without device allocation (for error-path testing).
// n<=0 cases are passed through directly to the API (returns SUCCESS without further validation).
inline aclblasStatus_t aclblasSswap_npu(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSswap(handle, n, x, incx, y, incy);
    }

    const size_t xBytes = static_cast<size_t>(n) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(n) * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dX);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    if (y != nullptr) {
        aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclblasStatus_t ret = aclblasSswap(handle, n, static_cast<float*>(dX), incx, static_cast<float*>(dY), incy);

    aclrtSynchronizeDevice();

    if (x != nullptr) {
        aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (y != nullptr) {
        aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
    return ret;
}

#endif // SSWAP_NPU_H
