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
* \file caxpy_host.cpp
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
#include "../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t K_FACTOR_4 = 4;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t MAX_DATA_COUNT = 38 * 1024 / sizeof(float);

struct CaxpyTilingData {
    uint32_t n;
    float alphaReal;
    float alphaImag;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

CaxpyTilingData CalTilingData(uint32_t n, uint32_t vecCoreNum, float alphaReal, float alphaImag)
{
    CaxpyTilingData tilingData;
    tilingData.n = n * COMPLEX_NUM;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t rowNumEachCore = n / vecCoreNum;
    uint32_t remainRowNum = n % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint32_t i = 0; i < remainRowNum; i++) {
            tilingData.calNum[i] = COMPLEX_NUM;
            tilingData.startOffset[i] = i * COMPLEX_NUM;
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            tilingData.calNum[i] = currNum * COMPLEX_NUM;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }

    return tilingData;
}

void GenMaskData(uint32_t *maskData)
{
    uint32_t offsetNum = COMPLEX_NUM;
    uint32_t complexCount = MAX_DATA_COUNT / COMPLEX_NUM;
    for (uint32_t i = 0; i < complexCount; i++) {
        maskData[offsetNum * i] = K_FACTOR_4 * i;
        maskData[offsetNum * i + 1] = K_FACTOR_4 * (i + complexCount);
    }
}

int aclblasCaxpy(aclblasHandle handle, const std::complex<float> *x, std::complex<float> *y,
                 const std::complex<float> alpha, const int64_t n, const int64_t incx, const int64_t incy)
{
    uint32_t numBlocks = DEFAULT_VECTOR_NUM;

    size_t totalByteSize = n * sizeof(std::complex<float>);
    int32_t deviceId = 0;

    aclrtStream stream = static_cast<aclrtStream>(handle);

    float alphaReal = alpha.real();
    float alphaImag = alpha.imag();

    CaxpyTilingData tiling = CalTilingData(n, numBlocks, alphaReal, alphaImag);

    uint32_t maskSize = MAX_DATA_COUNT * sizeof(uint32_t) * COMPLEX_NUM;

    uint8_t *xHost = reinterpret_cast<uint8_t *>(const_cast<std::complex<float> *>(x));
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *maskDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    uint32_t *maskHost = new uint32_t[MAX_DATA_COUNT * COMPLEX_NUM];
    GenMaskData(maskHost);

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CaxpyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CaxpyTilingData), &tiling, sizeof(CaxpyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    caxpy_kernel_do(xDevice, maskDevice, yDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(maskDevice);
    aclrtFree(tilingDevice);
    delete[] maskHost;

    return ACL_SUCCESS;
}