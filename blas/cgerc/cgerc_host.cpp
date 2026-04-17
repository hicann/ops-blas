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
 * \file cgerc_host.cpp
 * \brief Host side implementation for cgerc operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"

using aclblasHandle = void *;

#define GM_ADDR uint8_t*

extern void cgerc_kernel_do(GM_ADDR d_x, GM_ADDR d_y, GM_ADDR d_offset, GM_ADDR d_A, GM_ADDR work_space,
                            GM_ADDR tiling_gm, uint32_t num_blocks, void *stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;
constexpr uint32_t MAX_DATA_COUNT = 32 * 1024 / 4;
constexpr uint32_t NUM_FLAG = 2;
constexpr uint32_t NUM_INBYTES = 4;

// Tiling data structure
struct CgercTilingData {
    uint32_t m;
    uint32_t n;
    float alphaReal;
    float alphaImag;
    uint64_t startOffset[40];
    uint64_t calNum[40];
};

// Tiling calculation
CgercTilingData CalCgercTilingData(uint32_t m, uint32_t n, float alphaReal, float alphaImag, uint32_t vecCoreNum)
{
    CgercTilingData tilingData;
    memset(&tilingData, 0, sizeof(CgercTilingData));
    
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;
    
    // 按复数作tiling，以下单位都是复数
    uint64_t rowNumEachCore = m / vecCoreNum;
    uint64_t remainRowNum = m % vecCoreNum;
    
    if (rowNumEachCore == 0) {
        for (uint64_t i = 0; i < remainRowNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i;
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            tilingData.calNum[i] = currNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum;
        }
    }
    
    tilingData.m = m;
    tilingData.n = n;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;
    
    return tilingData;
}

// Create gather offset for cgerc
uint32_t* CreateCgercOffset()
{
    uint32_t gatherOffsetSize = MAX_DATA_COUNT;
    uint32_t *offsetData = nullptr;
    
    offsetData = new uint32_t[gatherOffsetSize];
    
    for (uint32_t i = 0; i < gatherOffsetSize / NUM_FLAG; i++) {
        offsetData[NUM_FLAG * i] = NUM_INBYTES * i;
        offsetData[NUM_FLAG * i + 1] = NUM_INBYTES * (i + gatherOffsetSize / NUM_FLAG);
    }
    
    return offsetData;
}

int aclblasCgerc(const int64_t m, const int64_t n,
                 const std::complex<float> &alpha,
                 const std::complex<float> *x, const int64_t incx,
                 const std::complex<float> *y, const int64_t incy,
                 std::complex<float> *A, const int64_t lda,
                 void *stream)
{
    // Extract alpha real and imaginary parts
    float alphaReal = alpha.real();
    float alphaImag = alpha.imag();
    
    uint32_t numBlocks = 8;  // Use all AICores
    
    CgercTilingData tiling = CalCgercTilingData(m, n, alphaReal, alphaImag, numBlocks);
    
    uint32_t *offset = CreateCgercOffset();
    
    // Calculate sizes
    // Complex numbers stored as [real, imag] pairs
    size_t xByteSize = m * sizeof(std::complex<float>);  // m complex numbers
    size_t yByteSize = n * sizeof(std::complex<float>);  // n complex numbers
    size_t AByteSize = m * n * sizeof(std::complex<float>);  // m x n complex matrix
    size_t offsetByteSize = MAX_DATA_COUNT * 2 * sizeof(uint32_t);
    size_t workspaceSize = 1024;
    
    uint8_t *xHost = reinterpret_cast<uint8_t *>(const_cast<std::complex<float> *>(x));
    uint8_t *yHost = reinterpret_cast<uint8_t *>(const_cast<std::complex<float> *>(y));
    uint8_t *AHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *offsetHost = reinterpret_cast<uint8_t *>(offset);
    
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *ADevice = nullptr;
    uint8_t *offsetDevice = nullptr;
    uint8_t *workspaceDevice = nullptr;
    uint8_t *tilingDevice = nullptr;
    
    // Allocate device memory
    aclrtMalloc((void **)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&ADevice, AByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offsetDevice, offsetByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CgercTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Copy inputs to device
    aclrtMemcpy(xDevice, xByteSize, xHost, xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, yByteSize, yHost, yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(ADevice, AByteSize, AHost, AByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, offsetByteSize, offsetHost, offsetByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CgercTilingData), &tiling, sizeof(CgercTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    
    // Launch kernel
    cgerc_kernel_do(xDevice, yDevice, offsetDevice, ADevice, workspaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);
    
    // Copy output back to host
    aclrtMemcpy(AHost, AByteSize, ADevice, AByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    // Free device memory
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(ADevice);
    aclrtFree(offsetDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);
    
    // Free host memory
    delete[] offset;
    
    return ACL_SUCCESS;
}
