/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "strttp_tiling_data.h"

void strttp_kernel_do(uint8_t* a, uint8_t* ap, uint8_t* tilingGm,
                      uint32_t numBlocks, void *stream);

// ---------------------------------------------------------------------------
// Parameter validation — returns non-success if invalid
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateParams(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, int lda, const float* A, const float* AP)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStrttp", "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        OP_LOGE("aclblasStrttp", "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        OP_LOGE("aclblasStrttp", "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasStrttp", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasStrttp", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasStrttp", "lda=%d, expected >= max(1,n)=%d", lda, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Query vector core count
// ---------------------------------------------------------------------------
static uint32_t GetVecCoreNum(int32_t deviceId, aclblasStatus_t* status)
{
    int64_t availableCoreNum = 0;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtGetDeviceInfo failed, error=%d", ret);
        *status = ACLBLAS_STATUS_EXECUTION_FAILED;
        return 0;
    }
    *status = ACLBLAS_STATUS_SUCCESS;
    return static_cast<uint32_t>(availableCoreNum);
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static aclblasStatus_t CalTilingData(TrttpTilingData* tiling, uint32_t n, uint32_t lda, uint32_t uploVal)
{
    int32_t deviceId = 0;
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtGetDevice failed, error=%d", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclblasStatus_t gStatus;
    uint32_t vecCoreNum = GetVecCoreNum(deviceId, &gStatus);
    if (gStatus != ACLBLAS_STATUS_SUCCESS) {
        return gStatus;
    }

    uint32_t useCoreNum = vecCoreNum;
    if (useCoreNum > n) {
        useCoreNum = n;
    }
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    tiling->n = n;
    tiling->lda = lda;
    tiling->uplo = uploVal;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel execution & cleanup
// ---------------------------------------------------------------------------
static aclblasStatus_t ExecuteKernel(const float* A, float* AP, const TrttpTilingData* tiling, aclrtStream stream)
{
    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(TrttpTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtMalloc failed, error=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(TrttpTilingData), tiling, sizeof(TrttpTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtMemcpy failed, error=%d", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    strttp_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(AP), tilingDevice,
        tiling->useCoreNum, stream);

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtSynchronizeStream failed, error=%d", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclRet = aclrtFree(tilingDevice);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasStrttp", "aclrtFree failed, error=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* A, int lda, float* AP)
{
    aclblasStatus_t status = ValidateParams(handle, n, uplo, lda, A, AP);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t uploVal = (uplo == ACLBLAS_LOWER) ? 0 : 1;
    uint32_t uN = static_cast<uint32_t>(n);
    uint32_t uLda = static_cast<uint32_t>(lda);

    TrttpTilingData tiling{};
    status = CalTilingData(&tiling, uN, uLda, uploVal);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return ExecuteKernel(A, AP, &tiling, h->stream);
}
