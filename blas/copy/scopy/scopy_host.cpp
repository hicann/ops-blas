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
#include "cann_ops_blas_common.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

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

aclblasStatus_t aclblasScopy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtStream useStream = nullptr;
    if (handle != nullptr) {
        auto* h = reinterpret_cast<_aclblas_handle*>(handle);
        useStream = h->stream;
    }

    uint32_t numBlocks = 8;
    CopyTilingData tiling = CalTilingData(n, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    scopy_kernel_do(x, y, nullptr, tilingDevice, numBlocks, useStream);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtStream useStream = nullptr;
    if (handle != nullptr) {
        auto* h = reinterpret_cast<_aclblas_handle*>(handle);
        useStream = h->stream;
    }

    uint32_t numBlocks = 8;
    uint64_t totalFloatNum = n * 2;
    CopyTilingData tiling = CalTilingData(totalFloatNum, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    scopy_kernel_do(x, y, nullptr, tilingDevice, numBlocks, useStream);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
