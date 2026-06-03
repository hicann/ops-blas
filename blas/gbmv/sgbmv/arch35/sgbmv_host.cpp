/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sgbmv_host.cpp
 * \brief single-precision gbmv host-side implementation
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgbmv_tiling_data.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static aclblasStatus_t ValidateSgbmvParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const float* alpha,
    const float* beta, const float* A, const float* x, const float* y)
{
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgbmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(kl >= 0, OP_LOGE("aclblasSgbmv", "invalid kl=%d", kl); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ku >= 0, OP_LOGE("aclblasSgbmv", "invalid ku=%d", ku); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        lda >= kl + ku + 1, OP_LOGE("aclblasSgbmv", "invalid lda=%d, kl=%d, ku=%d", lda, kl, ku);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSgbmv", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSgbmv", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSgbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSgbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSgbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
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

static SgbmvTilingData CalSgbmvTilingData(
    uint32_t useNumBlocks, int m, int n, int kl, int ku, int lda, aclblasOperation_t trans, float alpha, float beta,
    int incx, int incy)
{
    SgbmvTilingData tilingData{};
    bool isTransT = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTransT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(outDim, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.rowsPerBlock =
        static_cast<uint32_t>((outDim + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.kl = static_cast<uint32_t>(kl);
    tilingData.ku = static_cast<uint32_t>(ku);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.trans = isTransT ? 1U : 0U;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);

    return tilingData;
}

aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(
        m >= 0 && n >= 0, OP_LOGE("aclblasSgbmv", "invalid m=%d or n=%d", m, n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSgbmvParams(trans, m, n, kl, ku, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetVectorCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgbmv", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    bool isTransT = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTransT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(outDim, SIMT_MIN_THREAD_NUM), aivCoreNum);
    SgbmvTilingData tiling = CalSgbmvTilingData(useNumBlocks, m, n, kl, ku, lda, trans, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSgbmv", "tiling: m=%u n=%u kl=%u ku=%u lda=%u trans=%u numBlocks=%u numThreads=%u", tiling.m, tiling.n,
        tiling.kl, tiling.ku, tiling.lda, tiling.trans, useNumBlocks, tiling.numThreads);
    OP_LOGI("aclblasSgbmv", "launching kernel");

    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SgbmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSgbmv", "aclrtMalloc failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SgbmvTilingData), &tiling, sizeof(SgbmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSgbmv", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    sgbmv_kernel_do(
        (GM_ADDR) const_cast<float*>(A), (GM_ADDR) const_cast<float*>(x), (GM_ADDR)y, nullptr, tilingDevice,
        useNumBlocks, h->stream);

    aclRet = aclrtSynchronizeStream(h->stream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSgbmv", "aclrtSynchronizeStream failed, ret=%d", aclRet);
        aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
