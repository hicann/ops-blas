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
 * \file stpsv_host.cpp
 * \brief Single-precision tpsv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "stpsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"
#include "common/kernel_launch/aclblas_kernel_do.h"

void stpsv_kernel_do(
    GM_ADDR AP, GM_ADDR x, GM_ADDR tilingGm, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag,
    uint32_t numThreads, void* stream);

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static constexpr uint32_t SIMT_THRESHOLD = 128;

static aclblasStatus_t ValidateTpsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx, const float* AP,
    const float* x)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(AP != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CHECK_RET(h != nullptr, return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateTpsvParams(uplo, trans, diag, n, incx, AP, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Determine execution path: numThreads=0 → scalar, numThreads>0 → SIMT
    uint32_t numThreads = 0;
    if (static_cast<uint32_t>(n) >= SIMT_THRESHOLD) {
        numThreads = std::min(
            CeilAlign<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }

    StpsvTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int64_t>(incx);
    tiling.numThreads = numThreads;

    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(StpsvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(StpsvTilingData), &tiling, sizeof(StpsvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    stpsv_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float*>(AP)), reinterpret_cast<GM_ADDR>(x), tilingDevice,
        static_cast<aclblasFillMode>(uplo), static_cast<aclblasOperation>(trans),
        static_cast<aclblasDiagType>(diag), numThreads, h->stream);

    aclRet = aclrtSynchronizeStream(h->stream);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
