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
 * \file tpmv.asc
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t TILE_SIZE = 128;
constexpr uint32_t MAX_TILE_TASK = 8192;

struct tbmvTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t useCoreNum;
    int64_t incx;
    uint32_t tileSize;
    uint32_t tileRows;
    uint32_t taskCount;
    uint16_t taskBi[MAX_TILE_TASK];
    uint32_t taskStart[MAX_CORE_NUM];
    uint32_t taskStep[MAX_CORE_NUM];
};

tbmvTilingData CalTbmvTilingData(
    uint32_t totalRows, uint32_t totalDias, uint32_t lda, uint32_t vecCoreNum, int64_t incx)
{
    tbmvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.k = totalDias;
    tilingData.lda = lda;
    tilingData.incx = incx;
    tilingData.tileSize = TILE_SIZE;
    tilingData.tileRows = (totalRows + TILE_SIZE - 1) / TILE_SIZE;

    // Build 2D lower-triangle tile tasks: (bi, bj), bi >= bj.
    uint32_t taskCount = 0;
    for (uint32_t bi = 0; bi < totalDias; ++bi) {
        tilingData.taskBi[taskCount] = bi;
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
    if (tilingData.useCoreNum < vecCoreNum) {
        for (uint32_t i = tilingData.useCoreNum; i < vecCoreNum; ++i) {
            tilingData.taskStart[i] = -1;
        }
    }

    return tilingData;
}

aclblasStatus_t aclblasStbmv_legacy(
    aclblasHandle_t handle, const float* a, const int64_t lda, const float* x, float* y, const int64_t n,
    const int64_t k, const int64_t incx)
{
    const uint32_t rowCount = static_cast<uint32_t>(n);
    const uint32_t bandCount = static_cast<uint32_t>(k);
    const uint32_t leadingDim = static_cast<uint32_t>(lda);
    const size_t packedElementCount = static_cast<size_t>(rowCount) * (static_cast<size_t>(rowCount) + 1U) / 2U;

    std::vector<float> packedMatrix(packedElementCount, 0.0f);
    auto packedLowerIndex = [](uint32_t row, uint32_t col) {
        return static_cast<size_t>(col + (row * (row + 1U)) / 2U);
    };

    for (uint32_t row = 0; row < rowCount; ++row) {
        uint32_t startCol = row > bandCount ? row - bandCount : 0;
        for (uint32_t col = startCol; col <= row; ++col) {
            uint32_t bandRow = row - col;
            packedMatrix[packedLowerIndex(row, col)] = a[static_cast<size_t>(bandRow) * leadingDim + col];
        }
    }

    return aclblasStpmv_legacy(
        handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, packedMatrix.data(), x, y, incx);
}
