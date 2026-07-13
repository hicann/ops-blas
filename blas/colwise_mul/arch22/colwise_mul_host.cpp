/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License for the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file colwise_mul_host.cpp
 * \brief Host side implementation for colwise_mul operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void colwise_mul_kernel_do(uint8_t* mat, uint8_t* vec, uint8_t* aug, uint8_t* result,
                           uint8_t* workSpace, uint8_t* tilingGm,
                           uint32_t numBlocks, void *stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;
constexpr uint32_t MAX_DATA_COUNT = 32 * 1024 / sizeof(float);

// Tiling data structure
struct ColwiseMulTilingData {
    uint32_t m;
    uint32_t n;

    uint32_t startOffset[40];
    uint32_t calRowNum[40];
};

// Tiling calculation
ColwiseMulTilingData CalColwiseMulTilingData(uint32_t m, uint32_t n, uint32_t vecCoreNum)
{
    ColwiseMulTilingData tilingData;
    memset(&tilingData, 0, sizeof(ColwiseMulTilingData));

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    // n is already in float elements (for complex, it's 2 * num_complex)
    uint32_t rowNumEachCore = m / vecCoreNum;
    uint32_t remainRowNum = m % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint32_t i = 0; i < remainRowNum; i++) {
            tilingData.calRowNum[i] = 1;
            tilingData.startOffset[i] = n * i; // each row has n FP32 elements
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currRowNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currRowNum = rowNumEachCore + 1;
            } else {
                currRowNum = rowNumEachCore;
            }
            tilingData.calRowNum[i] = currRowNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currRowNum * n;
        }
    }

    tilingData.m = m; // num of rows
    tilingData.n = n; // num of FP32 elements each row

    return tilingData;
}

uint32_t* CreateAugColwiseMul()
{
    uint32_t complexCount = MAX_DATA_COUNT / COMPLEX_NUM;

    uint32_t* augData = nullptr;

    augData = new uint32_t[MAX_DATA_COUNT];

    for (uint32_t i = 0; i < complexCount; i++) {
        augData[COMPLEX_NUM * i] = FP32_BYTE_SIZE * i;
        augData[COMPLEX_NUM * i + 1] = FP32_BYTE_SIZE * (i + complexCount);
    }
    return augData;
}

aclblasStatus_t aclblasColwiseMul(
    aclblasHandle_t handle, const int64_t m, const int64_t n, uint8_t* mat, uint8_t* vec, uint8_t* result)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t nFloats = n * 2;
    uint32_t numBlocks = 8;

    ColwiseMulTilingData tiling = CalColwiseMulTilingData(m, nFloats, numBlocks);

    uint32_t* aug = CreateAugColwiseMul();
    size_t augByteSize = MAX_DATA_COUNT * sizeof(uint32_t);
    size_t workspaceSize = 1024;

    uint8_t* augDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&augDevice, augByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(augDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(ColwiseMulTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
        aclrtFree(augDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(augDevice, augByteSize, aug, augByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(augDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet = aclrtMemcpy(
        tilingDevice, sizeof(ColwiseMulTilingData), &tiling, sizeof(ColwiseMulTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(augDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    colwise_mul_kernel_do(mat, vec, augDevice, result, workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(augDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(augDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
