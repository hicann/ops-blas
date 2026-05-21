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
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "../utils/aclblas_kernel_do.h"
#include "../utils/aclblas_handle_internal.h"
#include "stpttr_tiling_data.h"

// ---------------------------------------------------------------------------
// Parameter validation — returns non-success if invalid
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateParams(aclblasHandle_t handle, int n, aclblasFillMode_t uplo,
                                      int lda, const float *AP, const float *A)
{
    if (handle == nullptr) {
        printf("[ERROR][tpttr] handle is nullptr\n");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        printf("[ERROR][tpttr] n=%d, expected >= 0\n", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        printf("[ERROR][tpttr] uplo=%d, expected 121(UPPER) or 122(LOWER)\n", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (lda < n) {
        printf("[ERROR][tpttr] lda=%d, expected >= n=%d\n", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (AP == nullptr) {
        printf("[ERROR][tpttr] AP must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        printf("[ERROR][tpttr] A must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Query vector core count
// ---------------------------------------------------------------------------
static uint32_t GetVecCoreNum(int32_t deviceId, aclblasStatus_t *status)
{
    int64_t availableCoreNum = 0;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtGetDeviceInfo failed, error=%d\n", ret);
        *status = ACLBLAS_STATUS_EXECUTION_FAILED;
        return 0;
    }
    *status = ACLBLAS_STATUS_SUCCESS;
    return static_cast<uint32_t>(availableCoreNum);
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static aclblasStatus_t CalTilingData(TpttrTilingData *tiling, int n, int lda, int uploVal)
{
    int32_t deviceId = 0;
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtGetDevice failed, error=%d\n", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclblasStatus_t gStatus;
    uint32_t vectorCoreNum = GetVecCoreNum(deviceId, &gStatus);
    if (gStatus != ACLBLAS_STATUS_SUCCESS) {
        return gStatus;
    }

    uint32_t useCoreNum = static_cast<uint32_t>(vectorCoreNum);
    if (useCoreNum > static_cast<uint32_t>(n)) {
        useCoreNum = static_cast<uint32_t>(n);
    }
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    tiling->uplo = uploVal;
    tiling->n = n;
    tiling->lda = lda;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel execution & cleanup
// ---------------------------------------------------------------------------
static aclblasStatus_t ExecuteKernel(const float *AP, float *A,
                                      const TpttrTilingData *tiling,
                                      aclrtStream stream)
{
    uint8_t *tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void **)&tilingDevice, sizeof(TpttrTilingData),
                                   ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtMalloc failed, error=%d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(tilingDevice, sizeof(TpttrTilingData),
                          tiling, sizeof(TpttrTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtMemcpy failed, error=%d\n", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    stpttr_kernel_do(reinterpret_cast<uint8_t *>(const_cast<float *>(AP)),
                     reinterpret_cast<uint8_t *>(A),
                     nullptr, tilingDevice, tiling->useCoreNum, stream);

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtSynchronizeStream failed, error=%d\n", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclRet = aclrtFree(tilingDevice);
    if (aclRet != ACL_SUCCESS) {
        printf("[ERROR][tpttr] aclrtFree failed, error=%d\n", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
aclblasStatus_t aclblastpttr(aclblasHandle_t handle, aclblasFillMode_t uplo,
                              int n, const float *AP, float *A, int lda)
{
    aclblasStatus_t status = ValidateParams(handle, n, uplo, lda, AP, A);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto *h = reinterpret_cast<_aclblas_handle *>(handle);
    int uploVal = (uplo == ACLBLAS_LOWER) ? 0 : 1;

    TpttrTilingData tiling{};
    status = CalTilingData(&tiling, n, lda, uploVal);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return ExecuteKernel(AP, A, &tiling, h->stream);
}
