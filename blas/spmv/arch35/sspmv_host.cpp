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
 * \file sspmv_host.cpp
 * \brief SSPMV Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "sspmv_tiling_data.h"

void sspmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static aclblasStatus_t ValidateSspmvParams(
    aclblasFillMode_t uplo, int incx, int incy, const float* alpha, const float* beta, const float* ap, const float* x,
    const float* y)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSspmv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSspmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSspmv", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSspmv", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspmv", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSspmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static uint32_t GetVectorCoreCount()
{
    int32_t deviceId = 0;
    int64_t vecCoreNum = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        return 0;
    }
    aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecCoreNum);
    return (vecCoreNum > 0) ? static_cast<uint32_t>(vecCoreNum) : 0;
}

static SspmvTilingData CalSspmvTilingData(
    uint32_t useNumBlocks, int n, aclblasFillMode_t uplo, float alpha, float beta, int incx, int incy)
{
    SspmvTilingData tilingData{};
    tilingData.nthreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);
    return tilingData;
}

aclblasStatus_t aclblasSspmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* ap, const float* x,
    int incx, const float* beta, float* y, int incy)
{
    // 1. n < 0 check
    CHECK_RET(n >= 0, OP_LOGE("aclblasSspmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    // 2. n == 0 quick return (must be before other validations)
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // 3. handle non-null check
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSspmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    // 4. parameter validation
    aclblasStatus_t st = ValidateSspmvParams(uplo, incx, incy, alpha, beta, ap, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    // 5. get vector core count
    uint32_t aivCoreNum = GetVectorCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSspmv", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);
    // 6. extract stream from handle
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;
    // 7. fill tiling data
    SspmvTilingData tiling = CalSspmvTilingData(useNumBlocks, n, uplo, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSspmv", "tiling: n=%u uplo=%u nthreads=%u numBlocks=%u", tiling.n, tiling.uplo, tiling.nthreads,
        useNumBlocks);
    OP_LOGI("aclblasSspmv", "launching kernel");

    // 8. allocate tiling device memory
    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SspmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSspmv", "aclrtMalloc failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);
    // 9. copy tiling to device
    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SspmvTilingData), &tiling, sizeof(SspmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSspmv", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    // 10. launch kernel
    sspmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(ap)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, tilingDevice,
        useNumBlocks, useStream);
    // 11. synchronize stream
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSspmv", "aclrtSynchronizeStream failed, ret=%d", aclRet);
        aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    // 12. cleanup
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
