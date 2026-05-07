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
 * \file scopy.asc
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

struct CopyTilingData {
    uint32_t n;
    uint32_t useCoreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

CopyTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum)
{
    CopyTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    // Set zero for startOffset and calNum
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }
    // Num of blocks
    uint32_t totalBlockNum = totalEleNum / ELEMENTS_PER_BLOCK_TILING;
    // Remain elements num
    uint32_t remainNum = totalEleNum % ELEMENTS_PER_BLOCK_TILING;

    if (totalBlockNum == 0) {
        // Use only 1 AIV core.
        tilingData.calNum[0] = remainNum;
        tilingData.useCoreNum = 1;
    } else if (totalBlockNum <= vecCoreNum) {
        for (uint32_t i = 0; i < totalBlockNum; i++) {
            tilingData.startOffset[i] = ELEMENTS_PER_BLOCK_TILING * i;
            tilingData.calNum[i] = ELEMENTS_PER_BLOCK_TILING;
        }
        tilingData.calNum[totalBlockNum - 1] += remainNum;
        tilingData.useCoreNum = totalBlockNum;
    } else {
        uint64_t blockNumEachCore;
        uint32_t remainBlock;

        blockNumEachCore = totalBlockNum / vecCoreNum;
        remainBlock = totalBlockNum % vecCoreNum;

        uint64_t currOffset = 0;
        uint64_t currCalNum = 0;

        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainBlock) {
                currCalNum = (blockNumEachCore + 1) * ELEMENTS_PER_BLOCK_TILING;
            } else {
                currCalNum = blockNumEachCore * ELEMENTS_PER_BLOCK_TILING;
            }
            tilingData.startOffset[i] = currOffset;
            tilingData.calNum[i] = currCalNum;
            currOffset += currCalNum;
        }
        tilingData.calNum[vecCoreNum - 1] += remainNum;
        tilingData.useCoreNum = vecCoreNum;
    }
    return tilingData;
}


int aclblasScopy(float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy, void *stream)
{
    uint32_t numBlocks = 8;

    size_t totalByteSize = n * sizeof(float);
    int32_t deviceId = 0;

    CopyTilingData tiling = CalTilingData(n, numBlocks);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // scopy_kernel<<<numBlocks, nullptr, stream>>>(xDevice, yDevice, nullptr, tilingDevice);
    scopy_kernel_do(xDevice, yDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);

    return ACL_SUCCESS;
}

int aclblasCcopy(std::complex<float> *x, std::complex<float> *y, const int64_t n, const int64_t incx, const int64_t incy, void *stream)
{
    uint32_t numBlocks = 8;
    uint64_t totalFloatNum = n * 2;
    size_t totalByteSize = totalFloatNum * sizeof(float);

    CopyTilingData tiling = CalTilingData(totalFloatNum, numBlocks);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    scopy_kernel_do(xDevice, yDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);

    return ACL_SUCCESS;
}

