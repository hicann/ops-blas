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
 * \file complex_mat_dot_host.cpp
 * \brief Complex matrix dot product host implementation
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

extern void complex_mat_dot_kernel_do(GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result,
                                      GM_ADDR tilingGm, uint32_t numBlocks, void *stream);

constexpr uint32_t COMPLEX_NUM = 2;

constexpr uint32_t MAX_DATA_COUNT = 27 * 1024 / sizeof(float);

constexpr uint32_t MUL_NUM = 2;

constexpr uint32_t FOUR_NUM = 4;
struct ComplexMatDotTilingData {
    uint32_t m;
    uint32_t n;
    uint64_t startOffset[40];
    uint32_t calNum[40];  // num in FP32 format
};

static void CalTilingData(ComplexMatDotTilingData &tilingData, uint32_t m, uint32_t n, uint32_t vecCoreNum)
{
    tilingData.m = m;
    tilingData.n = n;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    // Initialize arrays
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t numEachCore = m * n / vecCoreNum;  // num of complex
    uint32_t remainNum = m * n - vecCoreNum * numEachCore;

    if (numEachCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i * COMPLEX_NUM;  // each complex has 2 FP32 elements
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currNum = numEachCore + 1;
            } else {
                currNum = numEachCore;
            }
            tilingData.calNum[i] = currNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }
}

uint32_t* CreateAugComplexMatDot()
{
    uint32_t complexCount = MAX_DATA_COUNT / 2;

    uint32_t *augData = nullptr;

    augData = new uint32_t[MAX_DATA_COUNT];

    for (uint32_t i = 0; i < complexCount; i++) {
        augData[MUL_NUM * i] = FOUR_NUM * i;
        augData[MUL_NUM * i + 1] = FOUR_NUM * (i + complexCount);
    }
    return augData;
}

int aclblasComplexMatDot(const float *matx, const float *maty, float *result,
                         const int64_t m, const int64_t n, void *stream)
{
    uint32_t numBlocks = 8;

    // Each complex number has 2 floats (real and imaginary parts)
    size_t matrixByteSize = m * n * 2 * sizeof(float);  // 2*m*n floats for complex matrix
    size_t augByteSize = MAX_DATA_COUNT * sizeof(uint32_t);

    int32_t deviceId = 0;

    ComplexMatDotTilingData tiling;
    CalTilingData(tiling, m, n, numBlocks);
    uint32_t *aug = CreateAugComplexMatDot();

    uint8_t *matxHost = reinterpret_cast<uint8_t *>(const_cast<float *>(matx));
    uint8_t *matyHost = reinterpret_cast<uint8_t *>(const_cast<float *>(maty));
    uint8_t *augHost = reinterpret_cast<uint8_t *>(aug);
    uint8_t *resultHost = reinterpret_cast<uint8_t *>(result);
    uint8_t *matxDevice = nullptr;
    uint8_t *matyDevice = nullptr;
    uint8_t *augDevice = nullptr;
    uint8_t *resultDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&matxDevice, matrixByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&matyDevice, matrixByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&augDevice, augByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&resultDevice, matrixByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(ComplexMatDotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(matxDevice, matrixByteSize, matxHost, matrixByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(matyDevice, matrixByteSize, matyHost, matrixByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resultDevice, matrixByteSize, resultHost, matrixByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(augDevice, augByteSize, augHost, augByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(ComplexMatDotTilingData), &tiling, sizeof(ComplexMatDotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    complex_mat_dot_kernel_do(matxDevice, matyDevice, augDevice, resultDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(resultHost, matrixByteSize, resultDevice, matrixByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(matxDevice);
    aclrtFree(matyDevice);
    aclrtFree(augDevice);
    aclrtFree(resultDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
