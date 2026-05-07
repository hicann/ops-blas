/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use the file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
* \file cswap_host.cpp
* \brief
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t CACUL_TWO = 2;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;

struct CswapTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

static CswapTilingData CalCswapTilingData(uint32_t n, uint32_t vecCoreNum)
{
    CswapTilingData tilingData;
    tilingData.n = n;
    tilingData.coreNum = 0;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t complexNum = n;
    uint32_t numPerCore = complexNum / vecCoreNum;
    uint32_t remainNum = complexNum % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = CACUL_TWO;
            tilingData.startOffset[i] = CACUL_TWO * i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = (numPerCore + 1) * CACUL_TWO;
            } else {
                currCalNum = numPerCore * CACUL_TWO;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }

    return tilingData;
}

int aclblasCswap(aclblasHandle handle, float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy)
{
    uint32_t numBlocks = 8;

    size_t totalByteSize = n * 2 * sizeof(float);
    int32_t deviceId = 0;

    aclrtStream stream = static_cast<aclrtStream>(handle);

    CswapTilingData tiling = CalCswapTilingData(n, numBlocks);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CswapTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CswapTilingData), &tiling, sizeof(CswapTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    cswap_kernel_do(xDevice, yDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}