/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file syr_host.cpp
 * \brief syr Host-side dispatch: aclblasSsyr API with cyclic row distribution tiling.
 */

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "syr_tiling_data.h"

static SyrTilingData CalSyrTilingData(int64_t n, int64_t lda, uint32_t uplo, uint32_t coreNum)
{
    SyrTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = uplo;
    tiling.useCoreNum = 0;
    tiling.rowStride = 0;

    for (uint32_t i = 0; i < SYR_MAX_CORE_NUM; i++) {
        tiling.startRow[i] = 0;
        tiling.rowCount[i] = 0;
    }

    if (coreNum == 0) {
        coreNum = 1;
    }

    tiling.rowStride = coreNum;
    for (uint32_t i = 0; i < coreNum && i < SYR_MAX_CORE_NUM; i++) {
        tiling.startRow[i] = i;
        if (static_cast<uint32_t>(n) > i) {
            tiling.rowCount[i] = (static_cast<uint32_t>(n) - i + coreNum - 1) / coreNum;
        } else {
            tiling.rowCount[i] = 0;
        }
        if (tiling.rowCount[i] > 0) {
            tiling.useCoreNum = i + 1;
        }
    }

    return tiling;
}

namespace {

inline _aclblas_handle* ToInternal(aclblasHandle_t handle) { return reinterpret_cast<_aclblas_handle*>(handle); }

} // namespace

static aclblasStatus_t PrepareContiguousX(
    const float* x, int64_t n, int64_t incx, float** xContiguousDevice, bool* needFreeX)
{
    if (incx == 1) {
        *xContiguousDevice = const_cast<float*>(x);
        *needFreeX = false;
        return ACLBLAS_STATUS_SUCCESS;
    }

    size_t xContiguousBytes = static_cast<size_t>(n) * sizeof(float);
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(xContiguousDevice), xContiguousBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    *needFreeX = true;

    int64_t absIncx = std::abs(incx);
    size_t xTotalBytes = static_cast<size_t>(1 + (n - 1) * absIncx) * sizeof(float);
    std::vector<float> xHost(1 + (n - 1) * absIncx, 0.0f);
    aclRet = aclrtMemcpy(xHost.data(), xTotalBytes, const_cast<float*>(x), xTotalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyr", "aclrtMemcpy D2H failed, ret=%d", aclRet);
        aclrtFree(*xContiguousDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> xContiguousHost(n, 0.0f);
    if (incx > 0) {
        for (int64_t i = 0; i < n; i++) {
            xContiguousHost[i] = xHost[i * incx];
        }
    } else {
        // BLAS convention: negative incx means reversed storage.
        // x[0] is at memory offset (n-1)*absIncx, x[n-1] is at offset 0.
        for (int64_t i = 0; i < n; i++) {
            xContiguousHost[i] = xHost[(n - 1 - i) * absIncx];
        }
    }

    aclRet = aclrtMemcpy(
        *xContiguousDevice, xContiguousBytes, xContiguousHost.data(), xContiguousBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyr", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(*xContiguousDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t AllocAndCopyToDevice(uint8_t** devicePtr, const void* hostPtr, size_t size)
{
    aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(devicePtr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*devicePtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*devicePtr);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchSyrKernel(
    float alphaVal, const SyrTilingData& tiling, float* xContiguousDevice, bool needFreeX, float* A, aclrtStream stream)
{
    float* alphaDevice = nullptr;
    aclblasStatus_t status = AllocAndCopyToDevice(reinterpret_cast<uint8_t**>(&alphaDevice), &alphaVal, sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        if (needFreeX)
            aclrtFree(xContiguousDevice);
        return status;
    }

    uint8_t* tilingDevice = nullptr;
    status = AllocAndCopyToDevice(&tilingDevice, &tiling, sizeof(SyrTilingData));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(alphaDevice);
        if (needFreeX)
            aclrtFree(xContiguousDevice);
        return status;
    }

    syr_kernel_do(
        reinterpret_cast<GM_ADDR>(xContiguousDevice), reinterpret_cast<GM_ADDR>(A),
        reinterpret_cast<GM_ADDR>(alphaDevice), reinterpret_cast<GM_ADDR>(tilingDevice), tiling.useCoreNum, stream);

    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyr", "aclrtSynchronizeStream failed, ret=%d", aclRet);
        aclrtFree(alphaDevice);
        aclrtFree(tilingDevice);
        if (needFreeX)
            aclrtFree(xContiguousDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(alphaDevice);
    aclrtFree(tilingDevice);
    if (needFreeX) {
        aclrtFree(xContiguousDevice);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSsyr(
    aclblasHandle_t handle, aclblasFillMode uplo, const int n, const float* alpha, const float* x, const int incx,
    float* A, const int lda)
{
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (n < 0 || lda < std::max(1, n) || incx == 0 || alpha == nullptr || x == nullptr || A == nullptr) {
        OP_LOGE("aclblasSsyr", "invalid params: n=%d lda=%d incx=%d", n, lda, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    float alphaVal = *alpha;

    aclrtContext currentCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        OP_LOGE("aclblasSsyr", "aclrtGetCurrentContext failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        auto* h = ToInternal(handle);
        stream = h->stream;
    }

    uint32_t coreNum = (static_cast<uint32_t>(n) < SYR_MAX_CORE_NUM) ? static_cast<uint32_t>(n) : SYR_MAX_CORE_NUM;
    if (coreNum == 0) {
        coreNum = 1;
    }

    float* xContiguousDevice = nullptr;
    bool needFreeX = false;
    aclblasStatus_t status = PrepareContiguousX(x, n, incx, &xContiguousDevice, &needFreeX);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    SyrTilingData tiling = CalSyrTilingData(n, lda, (uplo == ACLBLAS_UPPER) ? 1u : 0u, coreNum);

    OP_LOGD(
        "aclblasSsyr", "tiling: n=%u lda=%u uplo=%u useCoreNum=%u", tiling.n, tiling.lda, tiling.uplo,
        tiling.useCoreNum);
    OP_LOGI("aclblasSsyr", "launching kernel");

    return LaunchSyrKernel(alphaVal, tiling, xContiguousDevice, needFreeX, A, stream);
}
