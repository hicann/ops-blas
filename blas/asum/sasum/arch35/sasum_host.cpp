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
 * \file sasum_host.cpp
 * \brief sasum Host-side dispatch for ascend950 
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "sasum_tiling_data.h"
#include "tiling/platform/platform_ascendc.h"

// arch35-style: tiling passed by value
void sasum_kernel_do(uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace,
                     const SasumTilingData& tiling, uint32_t numBlocks, void *stream);
// arch22-style: tiling passed as GM pointer (for backward compatibility)
void sasum_kernel_do(uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

namespace {

inline _aclblas_handle* ToInternal(aclblasHandle_t handle)
{
    return reinterpret_cast<_aclblas_handle*>(handle);
}

static uint32_t GetAivCoreNum()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclblasSasum", "PlatformAscendCManager::GetInstance() returned nullptr");
        return 0;
    }
    return platform->GetCoreNumAiv();
}

// AIV kernel internally handles 32B-alignment via maxDataCount tiling;
// SIMT kernel accesses elements individually — no block alignment needed here.
static SasumTilingData CalcSasumTilingData(int64_t totalEleNum, uint32_t vecCoreNum)
{
    SasumTilingData tiling;
    tiling.n = totalEleNum;
    tiling.useCoreNum = 0;

    for (uint32_t i = 0; i < SASUM_MAX_CORE_NUM; i++) {
        tiling.startOffset[i] = 0;
        tiling.calNum[i] = 0;
    }

    uint32_t useCoreNum = std::min(vecCoreNum, static_cast<uint32_t>(totalEleNum));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    tiling.useCoreNum = useCoreNum;

    uint32_t baseCount = static_cast<uint32_t>(totalEleNum) / useCoreNum;
    uint32_t remain    = static_cast<uint32_t>(totalEleNum) % useCoreNum;
    uint32_t offset    = 0;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        tiling.startOffset[i] = offset;
        tiling.calNum[i]      = baseCount + (i < remain ? 1 : 0);
        offset                += tiling.calNum[i];
    }

    return tiling;
}

static aclblasStatus_t ValidateSasumParams(aclblasHandle_t handle, int n,
                                            int incx, const float* x, float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSasum", "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        OP_LOGE("aclblasSasum", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        if (result != nullptr) {
            *result = 0.0f;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0) {
        OP_LOGE("aclblasSasum", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasSasum", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasSasum", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CalcSasumLaunchConfig(int n, int incx,
                                              uint32_t* numBlocks, uint32_t* nthreads)
{
    uint32_t aivCoreNum = GetAivCoreNum();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSasum", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    *numBlocks = (static_cast<uint32_t>(n) < aivCoreNum) ? static_cast<uint32_t>(n) : aivCoreNum;
    if (*numBlocks > SASUM_MAX_CORE_NUM) {
        *numBlocks = SASUM_MAX_CORE_NUM;
    }
    if (*numBlocks == 0) {
        *numBlocks = 1;
    }

    if (incx != 1) {
        *nthreads = std::min(
            CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), *numBlocks), SIMT_MIN_THREAD_NUM),
            SIMT_MAX_THREAD_NUM);
    } else {
        *nthreads = 0;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t SasumExecuteKernel(const float* x, float* result, int incx,
                                           uint32_t numBlocks, const SasumTilingData& tiling,
                                           aclrtStream stream)
{
    uint8_t* workspaceDevice = nullptr;
    if (incx != 1) {
        size_t workspaceBytes = static_cast<size_t>(numBlocks) * sizeof(float);
        aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice),
                                       workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasSasum", "aclrtMalloc workspace failed, size=%zu, ret=%d", workspaceBytes, aclRet);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }

    OP_LOGI("aclblasSasum", "launching kernel: blocks=%u", numBlocks);
    sasum_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
                    reinterpret_cast<uint8_t*>(result),
                    reinterpret_cast<uint8_t*>(workspaceDevice),
                    tiling, numBlocks, stream);

    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSasum", "aclrtSynchronizeStream failed, ret=%d", aclRet);
        if (workspaceDevice) aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (workspaceDevice) aclrtFree(workspaceDevice);
    return ACLBLAS_STATUS_SUCCESS;
}

}  // namespace

aclblasStatus_t aclblasSasum(aclblasHandle_t handle, int n,
                              const float* x, int incx, float* result)
{
    aclblasStatus_t status = ValidateSasumParams(handle, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto* h = ToInternal(handle);

    uint32_t numBlocks;
    uint32_t nthreads;
    status = CalcSasumLaunchConfig(n, incx, &numBlocks, &nthreads);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    SasumTilingData tiling = CalcSasumTilingData(n, numBlocks);
    tiling.incx = incx;
    tiling.nthreads = nthreads;

    OP_LOGD("aclblasSasum", "tiling: n=%ld incx=%ld useCoreNum=%u numBlocks=%u nthreads=%u",
            tiling.n, tiling.incx, tiling.useCoreNum, numBlocks, tiling.nthreads);

    return SasumExecuteKernel(x, result, incx, numBlocks, tiling, h->stream);
}
