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
 * \file ssymv_host.cpp
 * \brief SSYMV Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ssymv_tiling_data.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

template<typename R, typename T1, typename T2>
static inline R CeilDiv(T1 a, T2 b)
{
    R ra = static_cast<R>(a);
    R rb = static_cast<R>(b);
    return (ra + rb - 1) / rb;
}

template<typename R, typename T1, typename T2>
static inline R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}

static aclblasStatus_t ValidateSsymvParams(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy,
    const float* alpha, const float* beta, const float* A, const float* x, const float* y)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max(1, n), return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
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

static SsymvTilingData CalSsymvTilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, float alpha, float beta, int incx, int incy)
{
    SsymvTilingData tilingData{};
    tilingData.nthreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);
    return tilingData;
}

aclblasStatus_t aclblasSsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta,
    float* y, int incy)
{
    // 1. n < 0 check
    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    // 2. n == 0 quick return (must be before other validations)
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // 3. handle non-null check
    CHECK_RET(handle != nullptr, return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    // 4. parameter validation
    aclblasStatus_t st = ValidateSsymvParams(uplo, n, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    // 5. get vector core count
    uint32_t aivCoreNum = GetVectorCoreCount();
    if (aivCoreNum == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);
    // 6. extract stream from handle
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;
    // 7. fill tiling data
    SsymvTilingData tiling = CalSsymvTilingData(useNumBlocks, n, lda, uplo, *alpha, *beta, incx, incy);
    // 8. allocate tiling device memory
    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SsymvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return ACLBLAS_STATUS_ALLOC_FAILED);
    // 9. copy tiling to device
    aclRet = aclrtMemcpy(tilingDevice, sizeof(SsymvTilingData), &tiling, sizeof(SsymvTilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    // 10. launch kernel
    ssymv_kernel_do(
        (GM_ADDR) const_cast<float*>(A), (GM_ADDR) const_cast<float*>(x), (GM_ADDR)y, nullptr, tilingDevice,
        useNumBlocks, useStream);
    // 11. synchronize stream
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    // 12. cleanup
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
