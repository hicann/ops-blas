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
 * \file spmv.asc
 * \brief
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

extern void spmv_kernel_do(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workSpace, GM_ADDR tilingGm,
                            uint32_t numBlocks, void *stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t TILE_SIZE = 128;
constexpr uint32_t MAX_TILE_TASK = 8192;

struct SpmvTilingData {
    uint32_t n;
    uint32_t useCoreNum;
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
    uint32_t tileSize;
    uint32_t tileRows;
    uint32_t taskCount;
    uint16_t taskBi[MAX_TILE_TASK];
    uint32_t taskStart[MAX_CORE_NUM];
    uint32_t taskStep[MAX_CORE_NUM];
};

SpmvTilingData CalTilingData(uint32_t totalRows, uint32_t vecCoreNum, float alpha, float beta,
    int64_t incx, int64_t incy)
{
    SpmvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = incx;
    tilingData.incy = incy;
    tilingData.tileSize = TILE_SIZE;
    tilingData.tileRows = (totalRows + TILE_SIZE - 1) / TILE_SIZE;

    // Build 2D lower-triangle tile tasks: (bi, bj), bi >= bj.
    uint32_t taskCount = 0;
    for (uint32_t bi = 0; bi < totalRows; ++bi) {
        tilingData.taskBi[taskCount] = static_cast<uint16_t>(bi);
        ++taskCount;
    }
    tilingData.taskCount = taskCount;

    uint32_t availableCoreNum = vecCoreNum;
    if (availableCoreNum == 0) {
        availableCoreNum = 1;
    }
    if (availableCoreNum > MAX_CORE_NUM) {
        availableCoreNum = MAX_CORE_NUM;
    }
    tilingData.useCoreNum = std::min(taskCount, availableCoreNum);

    if (tilingData.useCoreNum == 0) {
        return tilingData;
    }

    for (uint32_t i = 0; i < tilingData.useCoreNum; ++i) {
        tilingData.taskStart[i] = i;
        tilingData.taskStep[i] = tilingData.useCoreNum;
    }

    if (tilingData.useCoreNum > vecCoreNum) {
        for (uint32_t i = tilingData.useCoreNum; i < vecCoreNum; ++i) {
            tilingData.taskStart[i] = -1;
        }
    }

    return tilingData;
}


int aclblasSpmv(const float *aPacked, const float *x, const float *y, float *z,
    const float alpha, const float beta,
    const int64_t n, const int64_t incx, const int64_t incy, void *stream)
{
    constexpr uint32_t numBlocks = 8;
    const size_t vecByteSize = static_cast<size_t>(n) * sizeof(float);
    const size_t nSize = static_cast<size_t>(n);
    const size_t packedEleNum = nSize * (nSize + 1U) / 2U;
    const size_t packedByteSize = packedEleNum * sizeof(float);

    SpmvTilingData tiling = CalTilingData(static_cast<uint32_t>(n), numBlocks, alpha, beta, incx, incy);

    uint8_t *aDevice = nullptr;
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *zDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&aDevice, packedByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&xDevice, vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&zDevice, vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(SpmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDevice, packedByteSize, aPacked, packedByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, vecByteSize, x, vecByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, vecByteSize, y, vecByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(SpmvTilingData), &tiling, sizeof(SpmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    spmv_kernel_do(aDevice, xDevice, yDevice, zDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(z, vecByteSize, zDevice, vecByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}

