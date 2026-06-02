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
 * \file stbmv_host.cpp
 * \brief single-precision triangular band matrix-vector multiply host implementation for ascend950
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "stbmv_common.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static constexpr bool STBMV_ENABLE_COLUMN_SIMD_FASTPATH = true;

static aclblasStatus_t ValidateStbmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const float* a, int lda,
    const float* x, int incx)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= k + 1, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(a != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
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

static StbmvTilingData CalStbmvTilingData(uint32_t useNumBlocks, int n, int k, int lda, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int incx)
{
    StbmvTilingData tilingData{};
    uint32_t nValue = static_cast<uint32_t>(n);
    uint32_t kValue = static_cast<uint32_t>(k);
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(nValue, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.numBlocks = useNumBlocks;
    tilingData.rowsPerBlock = CeilDiv<uint32_t>(nValue, useNumBlocks);
    tilingData.useUb =
        (incx == 1 && n >= 32 && tilingData.rowsPerBlock + kValue <= STBMV_UB_X_FLOATS) ? 1U : 0U;
    tilingData.n = nValue;
    tilingData.k = kValue;
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.trans = static_cast<uint32_t>(trans);
    tilingData.diag = static_cast<uint32_t>(diag);
    tilingData.incx = static_cast<int64_t>(incx);
    return tilingData;
}

static bool UseColumnSimdFastPath(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, int incx)
{
    return STBMV_ENABLE_COLUMN_SIMD_FASTPATH && (uplo == ACLBLAS_LOWER || uplo == ACLBLAS_UPPER) &&
           trans == ACLBLAS_OP_N && diag == ACLBLAS_NON_UNIT && n > 0 && k > 0 && incx == 1;
}

static StbmvFastTilingData CalStbmvFastTilingData(
    uint32_t aivCoreNum, int n, int k, int lda, aclblasFillMode_t uplo, uint32_t& fastBlocks)
{
    fastBlocks =
        std::min<uint32_t>(std::min<uint32_t>(static_cast<uint32_t>(k + 1), aivCoreNum), static_cast<uint32_t>(n));
    fastBlocks = std::max<uint32_t>(fastBlocks, 1);

    StbmvFastTilingData tilingData{};
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.useCoreNum = fastBlocks;
    return tilingData;
}

aclblasStatus_t aclblasStbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const float* A, int lda, float* x, int incx)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CHECK_RET(h != nullptr, return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateStbmvParams(uplo, trans, diag, n, k, A, lda, x, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (k == 0 && diag == ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetVectorCoreCount();
    CHECK_RET(aivCoreNum > 0, return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum);
    useNumBlocks = std::max<uint32_t>(useNumBlocks, 1);

    StbmvTilingData tilingData = CalStbmvTilingData(useNumBlocks, n, k, lda, uplo, trans, diag, incx);
    bool useFastPath = UseColumnSimdFastPath(uplo, trans, diag, n, k, incx);
    uint32_t fastBlocks = 1;
    StbmvFastTilingData fastTilingData{};
    if (useFastPath) {
        fastTilingData = CalStbmvFastTilingData(aivCoreNum, n, k, lda, uplo, fastBlocks);
    }

    uint8_t* workspaceDevice = nullptr;
    size_t workspaceSize = 0;
    if (k > 0) {
        workspaceSize = static_cast<size_t>(n) * sizeof(float);
        aclError aclRet =
            aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, return ACLBLAS_STATUS_ALLOC_FAILED);
    }

    if (useFastPath) {
        int kernelRet = stbmv_arch35_simd_fastpath_kernel_do(
            A, x, workspaceDevice, workspaceSize, tilingData, fastTilingData, fastBlocks, h->stream);
        CHECK_RET(kernelRet == ACL_SUCCESS, aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    } else {
        stbmv_arch35_kernel_do(A, x, workspaceDevice, tilingData, useNumBlocks, h->stream);
    }

    aclError aclRet = aclrtSynchronizeStream(h->stream);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
