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
 * \file strmv_host.cpp
 * \brief single-precision triangular matrix-vector multiply host implementation for ascend950
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "strmv_common.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static aclblasStatus_t ValidateStrmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float* a, int lda,
    const float* x, int incx)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max(1, n), return ACLBLAS_STATUS_INVALID_VALUE);
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

static StrmvTilingData CalStrmvTilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int incx)
{
    StrmvTilingData tilingData{};
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), useNumBlocks), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.trans = static_cast<uint32_t>(trans);
    tilingData.diag = static_cast<uint32_t>(diag);
    tilingData.incx = incx;
    return tilingData;
}

aclblasStatus_t aclblasStrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CHECK_RET(h != nullptr, return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateStrmvParams(uplo, trans, diag, n, A, lda, x, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetVectorCoreCount();
    CHECK_RET(aivCoreNum > 0, return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum);
    useNumBlocks = std::max<uint32_t>(useNumBlocks, 1);

    StrmvTilingData tilingData = CalStrmvTilingData(useNumBlocks, n, lda, uplo, trans, diag, incx);
    const size_t workspaceSize = static_cast<size_t>(n) * sizeof(float);
    uint8_t* workspaceDevice = nullptr;
    aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return ACLBLAS_STATUS_ALLOC_FAILED);

    strmv_arch35_kernel_do(A, x, workspaceDevice, tilingData, useNumBlocks, h->stream);

    aclRet = aclrtSynchronizeStream(h->stream);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
