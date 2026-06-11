/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in the compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sdot_host.cpp
 * \brief Real vector dot product: result = sum(x[i] * y[i])
 */

#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;

struct SdotTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t isconj;
    uint32_t startOffset[DEFAULT_VECTOR_NUM];
    uint32_t calNum[DEFAULT_VECTOR_NUM];
};

SdotTilingData CalSdotTilingData(uint32_t n, uint32_t vecCoreNum)
{
    SdotTilingData tilingData;
    tilingData.n = n;
    tilingData.isconj = 0;
    tilingData.coreNum = 0;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    if (vecCoreNum > DEFAULT_VECTOR_NUM) {
        vecCoreNum = DEFAULT_VECTOR_NUM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t numPerCore = n / vecCoreNum;
    uint32_t remainNum = n % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = numPerCore + 1;
            } else {
                currCalNum = numPerCore;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }
    return tilingData;
}

aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, const float* y, const int64_t incy,
    float* result)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    size_t workspaceSize = 1024;

    SdotTilingData tiling = CalSdotTilingData(static_cast<uint32_t>(n), numBlocks);

    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SdotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SdotTilingData), &tiling, sizeof(SdotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    sdot_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
                   reinterpret_cast<uint8_t*>(result), workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}