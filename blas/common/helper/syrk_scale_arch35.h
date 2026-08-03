/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

__aicore__ inline void SyrkScaleApply(
    AscendC::LocalTensor<float>& cUb, AscendC::LocalTensor<float>& cInUb,
    AscendC::LocalTensor<float>& tempUb,
    uint32_t offset, int32_t count, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    if (!skipTemp) {
        if (isBetaZero) {
            Muls(cUb[offset], tempUb[offset], alpha, count);
        } else {
            Muls(cUb[offset], cInUb[offset], beta, count);
            Axpy(cUb[offset], tempUb[offset], alpha, count);
        }
    } else {
        Muls(cUb[offset], cInUb[offset], beta, count);
    }
}

__aicore__ inline void SyrkScaleComputeResult(
    uint8_t uploMode, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols,
    uint32_t ubColStride, AscendC::LocalTensor<float>& cUb, AscendC::LocalTensor<float>& cInUb,
    AscendC::LocalTensor<float>& tempUb, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    bool uploUpper = (uploMode == ACLBLAS_UPPER);

    bool isFullInterior = uploUpper
        ? (jBase >= iBase + rows)
        : (jBase + cols <= iBase);

    if (isFullInterior) {
        int32_t totalCount = static_cast<int32_t>(ubColStride * cols);
        SyrkScaleApply(cUb, cInUb, tempUb, 0, totalCount, skipTemp, isBetaZero, alpha, beta);
        return;
    }

    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        uint32_t colOffset = c * ubColStride;

        uint32_t uploCount;
        if (uploUpper) {
            uploCount = (absJ >= iBase) ? Min(absJ - iBase + 1, rows) : 0;
        } else {
            uploCount = (absJ < iBase) ? rows : (rows - (absJ - iBase));
        }
        uint32_t nonUploCount = rows - uploCount;

        if (nonUploCount > 0 && uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(rows));
        }

        uint32_t computeCount = uploUpper ? uploCount : rows;
        if (computeCount > 0) {
            SyrkScaleApply(cUb, cInUb, tempUb, colOffset,
                static_cast<int32_t>(computeCount), skipTemp, isBetaZero, alpha, beta);
        }

        if (nonUploCount > 0 && !uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(nonUploCount));
        }
    }
}

template <typename Op>
__aicore__ inline void SyrkScaleProcess(
    Op& op, uint32_t rowStart, uint32_t rowEnd,
    uint32_t n, uint8_t uploMode, uint32_t scaleBlock)
{
    if (rowStart >= rowEnd) {
        return;
    }
    bool uploUpper = (uploMode == ACLBLAS_UPPER);

    for (uint32_t iBase = rowStart; iBase < rowEnd; iBase += scaleBlock) {
        uint32_t rows = Min<uint32_t>(scaleBlock, rowEnd - iBase);
        uint32_t iEnd = iBase + rows - 1;

        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? n : Min(iEnd + 1, n);

        for (uint32_t jBase = jStart; jBase < jLimit; jBase += scaleBlock) {
            uint32_t cols = Min<uint32_t>(scaleBlock, jLimit - jBase);
            op.ProcessBlock(iBase, jBase, rows, cols);
        }
    }
}
