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
 * \file symv.asc
 * \brief
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"

extern void symv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* workSpace, uint8_t* tilingGm,
    uint32_t numBlocks, void* stream);

constexpr uint32_t SYMV_MAX_CORE_NUM = 50;
constexpr uint32_t SYMV_TILE_SIZE = 128;
constexpr uint32_t SYMV_MAX_TILE_TASK = 4096;
constexpr uint32_t NUM_BLOCKS = 8;

struct SymvTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
    uint32_t tileSize;
    uint32_t tileRows;
    uint32_t taskCount;
    uint16_t taskBi[SYMV_MAX_TILE_TASK];
    uint16_t taskBj[SYMV_MAX_TILE_TASK];
    uint8_t taskType[SYMV_MAX_TILE_TASK];
    uint32_t taskStart[SYMV_MAX_CORE_NUM];
    uint32_t taskStep[SYMV_MAX_CORE_NUM];
};

static SymvTilingData CalSymvTilingData(
    uint32_t totalRows, uint32_t lda, uint32_t vecCoreNum, float alpha, float beta, int64_t incx, int64_t incy)
{
    SymvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.lda = lda;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = incx;
    tilingData.incy = incy;
    tilingData.tileSize = SYMV_TILE_SIZE;

    uint32_t taskCount = std::min(totalRows, SYMV_MAX_TILE_TASK);
    tilingData.taskCount = taskCount;

    uint32_t availableCoreNum = vecCoreNum == 0 ? 1U : vecCoreNum;
    if (availableCoreNum > SYMV_MAX_CORE_NUM) {
        availableCoreNum = SYMV_MAX_CORE_NUM;
    }
    tilingData.useCoreNum = std::min(taskCount, availableCoreNum);
    if (tilingData.useCoreNum == 0) {
        return tilingData;
    }

    for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
        tilingData.taskBi[taskIdx] = static_cast<uint16_t>(taskIdx);
        tilingData.taskBj[taskIdx] = 0;
        tilingData.taskType[taskIdx] = 0;
    }

    for (uint32_t i = 0; i < tilingData.useCoreNum; ++i) {
        tilingData.taskStart[i] = i;
        tilingData.taskStep[i] = tilingData.useCoreNum;
    }
    return tilingData;
}

aclblasStatus_t aclblasSsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    (void)uplo;
    aclrtStream useStream = nullptr;
    if (handle != nullptr) {
        auto* h = reinterpret_cast<_aclblas_handle*>(handle);
        useStream = h->stream;
    }
    constexpr uint32_t numBlocks = 8;
    const size_t vecElementCount = static_cast<size_t>(n);
    const size_t matrixElementCount = static_cast<size_t>(n) * static_cast<size_t>(lda);
    const size_t vecByteSize = vecElementCount * sizeof(float);
    const size_t matrixByteSize = matrixElementCount * sizeof(float);

    SymvTilingData tiling = CalSymvTilingData(
        static_cast<uint32_t>(n), static_cast<uint32_t>(lda), numBlocks, *alpha, *beta, incx, incy);

    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* zDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc(reinterpret_cast<void**>(&aDevice), matrixByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&xDevice), vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&yDevice), vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&zDevice), vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(SymvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDevice, matrixByteSize, a, matrixByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, vecByteSize, x, vecByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, vecByteSize, y, vecByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(SymvTilingData), &tiling, sizeof(SymvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    symv_kernel_do(aDevice, xDevice, yDevice, zDevice, nullptr, tilingDevice, numBlocks, useStream);
    aclrtSynchronizeStream(useStream);
    aclrtMemcpy(y, vecByteSize, zDevice, vecByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
