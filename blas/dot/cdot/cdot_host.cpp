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
* \file cdot_host.cpp
* \brief Complex dot product: result = conj(x) ? dot(conj(x), y) : dot(x, y)
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"

using aclblasHandle = void *;

#define GM_ADDR uint8_t*

extern void cdot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm,
                           uint32_t numBlocks, void *stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;

struct CdotTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t isConj;
    uint32_t startOffset[DEFAULT_VECTOR_NUM];
    uint32_t calNum[DEFAULT_VECTOR_NUM];
};

CdotTilingData CalTilingData(uint32_t n, uint32_t vecCoreNum, uint32_t isConj)
{
    CdotTilingData tilingData;
    tilingData.n = n;
    tilingData.isConj = isConj;
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

    uint32_t complexNum = n / COMPLEX_NUM;
    uint32_t numPerCore = complexNum / vecCoreNum;
    uint32_t remainNum = complexNum % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = COMPLEX_NUM;
            tilingData.startOffset[i] = COMPLEX_NUM * i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = (numPerCore + 1) * COMPLEX_NUM;
            } else {
                currCalNum = numPerCore * COMPLEX_NUM;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }
    return tilingData;
}

int aclblasCdot(const float *x, const float *y, float *result,
                const int64_t n, const int64_t isConj, void *stream)
{
    uint32_t numBlocks = 8;

    size_t totalByteSize = 2 * n * sizeof(float);
    size_t workspaceSize = 1024;

    CdotTilingData tiling = CalTilingData(static_cast<uint32_t>(n * 2), numBlocks, static_cast<uint32_t>(isConj));

    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *resultDevice = nullptr;
    uint8_t *workspaceDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&resultDevice, 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CdotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, x, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, y, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CdotTilingData), &tiling, sizeof(CdotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    cdot_kernel_do(xDevice, yDevice, resultDevice, workspaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(result, 2 * sizeof(float), resultDevice, 2 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(resultDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
