/**
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

#define GM_ADDR uint8_t*

void strsv_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm,
                            aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag,
                            int64_t n, int64_t lda,
                            uint32_t numBlocks, void* stream);

constexpr uint32_t MAX_CORE_NUM = 40;

struct StrsvTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t startRow[MAX_CORE_NUM];
    uint32_t rowCount[MAX_CORE_NUM];
};

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

static StrsvTilingData CalStrsvTilingData(int64_t n, int64_t lda, uint32_t coreNum)
{
    StrsvTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.useCoreNum = 0;

    for (uint32_t i = 0; i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = 0;
        tiling.rowCount[i] = 0;
    }

    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t totalBlockNum = coreNum;
    uint32_t rowsPerBlock = (n + totalBlockNum - 1) / totalBlockNum;

    for (uint32_t i = 0; i < totalBlockNum && i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = i * rowsPerBlock;
        uint32_t rowEnd = (i + 1) * rowsPerBlock;
        if (rowEnd > n) rowEnd = n;
        tiling.rowCount[i] = (rowEnd > tiling.startRow[i]) ? (rowEnd - tiling.startRow[i]) : 0;

        if (tiling.rowCount[i] > 0) {
            tiling.useCoreNum = i + 1;
        }
    }

    return tiling;
}

int aclblasStrsv(aclblasHandle handle,
                 aclblasFillMode uplo,
                 aclblasOperation trans,
                 aclblasDiagType diag,
                 int64_t n,
                 const float *A,
                 int64_t lda,
                 float *x,
                 int64_t incx)
{
    if (n <= 0) {
        return ACL_SUCCESS;
    }

    if (A == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    int32_t deviceId = 0;
    aclrtContext currentCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    size_t aSize = static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(float);
    size_t xSize = static_cast<size_t>(n) * static_cast<size_t>(std::abs(incx)) * sizeof(float);

    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&xDevice), xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    std::vector<float> xHost(n, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        xHost[i] = x[i * incx];
    }

    aclRet = aclrtMemcpy(aDevice, aSize, A, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(xDevice, xSize, xHost.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    constexpr uint32_t numBlocks = 1;

    StrsvTilingData tiling = CalStrsvTilingData(n, lda, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(StrsvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(tilingDevice, sizeof(StrsvTilingData), &tiling, sizeof(StrsvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    strsv_kernel_do(aDevice, xDevice, tilingDevice, uplo, trans, diag, n, lda, tiling.useCoreNum, handle);

    aclRet = aclrtSynchronizeStream(handle);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(xHost.data(), xSize, xDevice, xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    for (int64_t i = 0; i < n; i++) {
        x[i * incx] = xHost[i];
    }

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}