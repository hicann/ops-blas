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

static aclblasStatus_t SrotmAllocCopyInputs(
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

inline aclblasStatus_t aclblasSrotm_npu(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSrotm(handle, n, x, incx, y, incy, param);
    }

    const size_t xBytes = static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((n - 1) * std::abs(incy) + 1) * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    aclblasStatus_t allocRet = SrotmAllocCopyInputs(x, xBytes, y, yBytes, &dX, &dY);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        return allocRet;
    }

    aclblasStatus_t ret = aclblasSrotm(handle, n, static_cast<float*>(dX), incx, static_cast<float*>(dY), incy, param);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // x/y 回拷彼此独立执行：任一失败都如实记录，不因 x 失败而跳过 y
    aclError xCopyRet = ACL_SUCCESS;
    aclError yCopyRet = ACL_SUCCESS;
    if (x != nullptr && dX != nullptr) {
        xCopyRet = aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (y != nullptr && dY != nullptr) {
        yCopyRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && (xCopyRet != ACL_SUCCESS || yCopyRet != ACL_SUCCESS)) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (dX != nullptr) {
        aclrtFree(dX);
    }
    if (dY != nullptr) {
        aclrtFree(dY);
    }
    return ret;
}
