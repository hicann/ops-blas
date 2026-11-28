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

static aclblasStatus_t SdotAllocCopyInputs(
    const float* x, size_t xBytes, const float* y, size_t yBytes, void** dX, void** dY)
{
    if (x != nullptr) {
        aclError aclRet = aclrtMalloc(dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(*dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(*dX);
            *dX = nullptr;
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    if (y != nullptr) {
        aclError aclRet = aclrtMalloc(dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (*dX != nullptr) {
                aclrtFree(*dX);
                *dX = nullptr;
            }
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(*dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (*dX != nullptr) {
                aclrtFree(*dX);
                *dX = nullptr;
            }
            aclrtFree(*dY);
            *dY = nullptr;
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSdot_npu(
    aclblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSdot(handle, n, x, incx, y, incy, result);
    }

    const size_t xBytes = (n > 0) ? static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(float) : sizeof(float);
    const size_t yBytes = (n > 0) ? static_cast<size_t>((n - 1) * std::abs(incy) + 1) * sizeof(float) : sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    aclblasStatus_t allocRet = SdotAllocCopyInputs(x, xBytes, y, yBytes, &dX, &dY);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        return allocRet;
    }

    void* dResult = nullptr;
    aclError aclRet = aclrtMalloc(&dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasSdot(
        handle, n, static_cast<const float*>(dX), incx, static_cast<const float*>(dY), incy,
        static_cast<float*>(dResult));

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        aclrtFree(dResult);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // 回拷结果先以独立标志记录，仅在内核调用成功时映射为 INTERNAL_ERROR，保留具体错误码
    aclError resultCopyRet = ACL_SUCCESS;
    if (result != nullptr) {
        resultCopyRet = aclrtMemcpy(result, sizeof(float), dResult, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && resultCopyRet != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(dX);
    aclrtFree(dY);
    aclrtFree(dResult);
    return ret;
}
