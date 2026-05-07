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
* \file sscal_host.cpp
* \brief
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <complex>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

struct SscalTilingData {
    uint32_t n;
    uint32_t useCoreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
    float alpha;
};

SscalTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum, float alpha)
{
    SscalTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;
    tilingData.alpha = alpha;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t totalBlockNum = totalEleNum / ELEMENTS_PER_BLOCK_TILING;
    uint32_t remainNum = totalEleNum % ELEMENTS_PER_BLOCK_TILING;

    if (totalBlockNum == 0) {
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


int aclblasSscal(aclblasHandle handle, float *x, const float alpha, const int64_t n, const int64_t incx)
{
    uint32_t numBlocks = 8;

    size_t totalByteSize = n * sizeof(float);
    int32_t deviceId = 0;

    aclrtStream stream = static_cast<aclrtStream>(handle);

    SscalTilingData tiling = CalTilingData(n, numBlocks, alpha);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *xDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    sscal_kernel_do(xDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}

int aclblasCsscal(aclblasHandle handle, std::complex<float> *x, const float alpha, const int64_t n, const int64_t incx)
{
    uint32_t numBlocks = 8;

    uint32_t totalFloatNum = n * 2;
    size_t totalByteSize = n * sizeof(std::complex<float>);
    int32_t deviceId = 0;

    aclrtStream stream = static_cast<aclrtStream>(handle);

    SscalTilingData tiling = CalTilingData(totalFloatNum, numBlocks, alpha);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *xDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    sscal_kernel_do(xDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}