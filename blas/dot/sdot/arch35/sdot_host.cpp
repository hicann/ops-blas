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
 * \file sdot_host.cpp
 * \brief Single-precision dot product (SIMD AIV): result = sum(x[i] * y[i])
 *        Arch35 (ascend950) host-side implementation.
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sdot_tiling_data.h"

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateSdotParams(
    aclblasHandle_t handle, int64_t incx, int64_t incy, const float* x, const float* y, const float* result)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSdot", "handle is nullptr"); return ACLBLAS_STATUS_NOT_INITIALIZED);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSdot", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSdot", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSdot", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSdot", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        result != nullptr, OP_LOGE("aclblasSdot", "result must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Query AI Vector Core count
// ---------------------------------------------------------------------------
static uint32_t GetVectorCoreCount()
{
    int32_t deviceId = 0;
    int64_t vecCoreNum = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "aclrtGetDevice failed");
        return 0;
    }
    aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecCoreNum);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "aclrtGetDeviceInfo failed, ret=%d", ret);
        return 0;
    }
    return (vecCoreNum > 0) ? static_cast<uint32_t>(vecCoreNum) : 0;
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static SdotTilingData CalSdotTilingData(int64_t n, int64_t incx, int64_t incy, uint32_t vectorCoreNum)
{
    SdotTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;

    uint32_t useCoreNum = std::min(vectorCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    if (useCoreNum > SDOT_MAX_CORE_NUM) {
        useCoreNum = SDOT_MAX_CORE_NUM;
    }
    tiling.useCoreNum = useCoreNum;

    uint32_t baseCount = static_cast<uint32_t>(n) / useCoreNum;
    uint32_t remain = static_cast<uint32_t>(n) % useCoreNum;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        tiling.startOffset[i] = offset;
        tiling.calCount[i] = baseCount + (i < remain ? 1 : 0);
        offset += tiling.calCount[i];
    }
    for (uint32_t i = useCoreNum; i < SDOT_MAX_CORE_NUM; i++) {
        tiling.startOffset[i] = 0;
        tiling.calCount[i] = 0;
    }

    return tiling;
}

// ---------------------------------------------------------------------------
// Kernel launch helper
// ---------------------------------------------------------------------------
static aclblasStatus_t LaunchSdotKernel(
    const SdotTilingData& tiling, const float* x, const float* y, float* result, aclrtStream stream)
{
    uint32_t workspaceSize = tiling.useCoreNum * sizeof(float);
    uint8_t* workspaceDevice = nullptr;
    aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblasSdot", "aclrtMalloc workspace failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    uint8_t* tilingDevice = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SdotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "aclrtMalloc tiling failed, ret=%d", aclRet);
        aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SdotTilingData), &tiling, sizeof(SdotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    OP_LOGI("aclblasSdot", "launching kernel with %u cores", tiling.useCoreNum);

    sdot_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float*>(x)), reinterpret_cast<GM_ADDR>(const_cast<float*>(y)),
        reinterpret_cast<GM_ADDR>(result), workspaceDevice, tilingDevice, tiling.useCoreNum, stream);

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "aclrtSynchronizeStream failed, ret=%d", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclrtFree(tilingDevice);
    aclrtFree(workspaceDevice);
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API: aclblasSdot
// ---------------------------------------------------------------------------
aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, const float* y, const int64_t incy,
    float* result)
{
    if (n <= 0) {
        if (result != nullptr) {
            aclError ret = aclrtMemset(result, sizeof(float), 0, sizeof(float));
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSdot", "aclrtMemset failed, ret=%d", ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateSdotParams(handle, incx, incy, x, y, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);

    uint32_t vectorCoreNum = GetVectorCoreCount();
    if (vectorCoreNum == 0) {
        OP_LOGE("aclblasSdot", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    SdotTilingData tiling = CalSdotTilingData(n, incx, incy, vectorCoreNum);

    OP_LOGD(
        "aclblasSdot", "tiling: n=%ld incx=%ld incy=%ld useCoreNum=%u", tiling.n, tiling.incx, tiling.incy,
        tiling.useCoreNum);

    return LaunchSdotKernel(tiling, x, y, result, h->stream);
}
