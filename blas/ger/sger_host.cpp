/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sger.asc
 * \brief SGER operation implementation with multi-block parallelism
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define GM_ADDR uint8_t*

extern void sger_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR alpha, GM_ADDR tilingGm,
                          uint32_t numBlocks, void* stream);



constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t MAX_CORE_NUM = 50;

struct SgerTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t startRow[MAX_CORE_NUM];
    uint32_t rowCount[MAX_CORE_NUM];
    uint32_t startCol[MAX_CORE_NUM];
    uint32_t colCount[MAX_CORE_NUM];
};

static SgerTilingData CalSgerTilingData(int64_t m, int64_t n, int64_t lda, uint32_t coreNum)
{
    SgerTilingData tiling;
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.useCoreNum = 0;


    for (uint32_t i = 0; i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = 0;
        tiling.rowCount[i] = 0;
        tiling.startCol[i] = 0;
        tiling.colCount[i] = 0;
    }

    if (coreNum == 0) {
        coreNum = 1;
    }


    uint32_t totalBlockNum = coreNum;
    uint32_t rowsPerBlock = (m + totalBlockNum - 1) / totalBlockNum;

    for (uint32_t i = 0; i < totalBlockNum && i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = i * rowsPerBlock;
        uint32_t rowEnd = (i + 1) * rowsPerBlock;
        if (rowEnd > m) rowEnd = m;
        tiling.rowCount[i] = (rowEnd > tiling.startRow[i]) ? (rowEnd - tiling.startRow[i]) : 0;


        tiling.startCol[i] = 0;
        tiling.colCount[i] = tiling.rowCount[i] > 0 ? n : 0;

        if (tiling.rowCount[i] > 0) {
            tiling.useCoreNum = i + 1;
        }
    }

    return tiling;
}

int aclblasSger(aclblasHandle handle, int64_t m, int64_t n, const float* alpha,
               const float* x, int64_t incx,
               float* y, int64_t incy,
               float* A, int64_t lda,
               void* stream)
{
    if (m <= 0 || n <= 0) {
        return ACL_SUCCESS;
    }

    if (alpha == nullptr || x == nullptr || A == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtContext currentCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }


    size_t aSize = static_cast<size_t>(m) * static_cast<size_t>(lda) * sizeof(float);
    size_t xSize = static_cast<size_t>(m) * static_cast<size_t>(std::abs(incx)) * sizeof(float);
    size_t ySize = static_cast<size_t>(n) * static_cast<size_t>(std::abs(incy)) * sizeof(float);
    size_t alphaSize = sizeof(float);

    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* alphaDevice = nullptr;

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&xDevice), xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&yDevice), ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&alphaDevice), alphaSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }


    std::vector<float> aHost(m * lda, 0.0f);
    std::vector<float> xHost(m, 0.0f);
    std::vector<float> yHost(n, 0.0f);

    for (int64_t i = 0; i < m; i++) {
        xHost[i] = x[i * incx];
    }

    for (int64_t j = 0; j < n; j++) {
        yHost[j] = y[j * incy];
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            aHost[i * lda + j] = A[i * lda + j];
        }
    }

    aclRet = aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(xDevice, xSize, xHost.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(yDevice, ySize, yHost.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(alphaDevice, alphaSize, alpha, alphaSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    constexpr uint32_t numBlocks = 8;


    SgerTilingData tiling = CalSgerTilingData(m, n, lda, numBlocks);


    uint8_t* tilingDevice = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SgerTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }


    aclRet = aclrtMemcpy(tilingDevice, sizeof(SgerTilingData), &tiling, sizeof(SgerTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }


    sger_kernel_do(aDevice, xDevice, yDevice, alphaDevice, tilingDevice,
                  tiling.useCoreNum, stream);

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(aHost.data(), aSize, aDevice, aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(alphaDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A[i * lda + j] = aHost[i * lda + j];
        }
    }

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(alphaDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
