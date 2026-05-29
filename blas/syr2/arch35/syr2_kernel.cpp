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
 * \file syr2_kernel.cpp
 * \brief Single-precision symmetric rank-2 update.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "syr2_tiling_data.h"
#include "common/helper/kernel_constant.h"

static constexpr uint32_t UB_HALF_FLOATS = 8192; // 32 KB per vector (x / y)

template <bool UPLO_IS_UPPER>
__simt_callee__ inline void GetColRange(uint32_t row, uint32_t n, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (UPLO_IS_UPPER) {
        colStart = row;
        colEnd = n;
    } else {
        colStart = 0;
        colEnd = row + 1;
    }
}

// ==========================================================================
//  GM path — grid-stride (incx != 1 or incy != 1 or small n)
// ==========================================================================
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Ssyr2Gm(
    uint32_t n, uint32_t lda, float alpha, int64_t incx, int64_t incy, __gm__ const float* xGm, __gm__ const float* yGm,
    __gm__ float* aGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float xRow = (incx >= 0) ? xGm[row * incx] : xGm[(n - 1 - row) * (-incx)];
        float yRow = (incy >= 0) ? yGm[row * incy] : yGm[(n - 1 - row) * (-incy)];
        float axRow = alpha * xRow;
        float ayRow = alpha * yRow;

        uint32_t colStart, colEnd;
        GetColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float xCol = (incx >= 0) ? xGm[col * incx] : xGm[(n - 1 - col) * (-incx)];
            float yCol = (incy >= 0) ? yGm[col * incy] : yGm[(n - 1 - col) * (-incy)];
            uint64_t aIdx = static_cast<uint64_t>(row) * lda + col;
            aGm[aIdx] = aGm[aIdx] + axRow * yCol + ayRow * xCol;
        }
    }
}

__aicore__ inline void FallbackToGm(
    const __gm__ Syr2TilingData* __restrict tdata, __gm__ float* __restrict xGm, __gm__ float* __restrict yGm,
    __gm__ float* __restrict aGm)
{
    if (tdata->uplo == ACLBLAS_UPPER) {
        asc_vf_call<Ssyr2Gm<true>>(
            dim3{tdata->numThreads, 1, 1}, tdata->n, tdata->lda, tdata->alpha, tdata->incx, tdata->incy, xGm, yGm, aGm);
    } else {
        asc_vf_call<Ssyr2Gm<false>>(
            dim3{tdata->numThreads, 1, 1}, tdata->n, tdata->lda, tdata->alpha, tdata->incx, tdata->incy, xGm, yGm, aGm);
    }
}

// ==========================================================================
//  UB path — x and y cached in __ubuf__ shared memory (incx == 1, incy == 1)
// ==========================================================================
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Ssyr2Ub(
    uint32_t n, uint32_t lda, float alpha, __gm__ const float* xGm, __gm__ const float* yGm, __gm__ float* aGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t xBase, uint32_t xLen, uint32_t yBase, uint32_t yLen)
{
    __ubuf__ float xUb[UB_HALF_FLOATS];
    __ubuf__ float yUb[UB_HALF_FLOATS];

    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x)
        xUb[i] = xGm[xBase + i];

    for (uint32_t i = threadIdx.x; i < yLen; i += blockDim.x)
        yUb[i] = yGm[yBase + i];

    asc_syncthreads();

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        float xRow = xUb[row - xBase];
        float yRow = yUb[row - yBase];
        float axRow = alpha * xRow;
        float ayRow = alpha * yRow;

        uint32_t colStart, colEnd;
        GetColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float xCol = xUb[col - xBase];
            float yCol = yUb[col - yBase];
            uint64_t aIdx = static_cast<uint64_t>(row) * lda + col;
            aGm[aIdx] = aGm[aIdx] + axRow * yCol + ayRow * xCol;
        }
    }
}

// ==========================================================================
//  Kernel dispatcher — choose GM or UB path per launch
// ==========================================================================
__global__ __aicore__ void syr2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const __gm__ Syr2TilingData* __restrict tdata =
        reinterpret_cast<const __gm__ Syr2TilingData* __restrict>(tilingGm);
    auto* __restrict xGm = reinterpret_cast<__gm__ float* __restrict>(x);
    auto* __restrict yGm = reinterpret_cast<__gm__ float* __restrict>(y);
    auto* __restrict aGm = reinterpret_cast<__gm__ float* __restrict>(A);

    if (tdata->incx != 1 || tdata->incy != 1 || tdata->n < 32) {
        FallbackToGm(tdata, xGm, yGm, aGm);
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowsPerBlk = tdata->rowsPerBlock;
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    uint32_t rowEnd = (rowStart + rowsPerBlk < tdata->n) ? (rowStart + rowsPerBlk) : tdata->n;
    if (rowStart >= rowEnd) {
        return;
    }

    uint32_t xBase = (tdata->uplo == ACLBLAS_UPPER) ? rowStart : 0;
    uint32_t xLen = (tdata->uplo == ACLBLAS_UPPER) ? (tdata->n - rowStart) : rowEnd;

    if (xLen > UB_HALF_FLOATS) {
        FallbackToGm(tdata, xGm, yGm, aGm);
        return;
    }

    if (tdata->uplo == ACLBLAS_UPPER) {
        asc_vf_call<Ssyr2Ub<true>>(
            dim3{tdata->numThreads, 1, 1}, tdata->n, tdata->lda, tdata->alpha, xGm, yGm, aGm, rowStart, rowEnd, xBase,
            xLen, xBase, xLen);
    } else {
        asc_vf_call<Ssyr2Ub<false>>(
            dim3{tdata->numThreads, 1, 1}, tdata->n, tdata->lda, tdata->alpha, xGm, yGm, aGm, rowStart, rowEnd, xBase,
            xLen, xBase, xLen);
    }
}

void syr2_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    syr2_kernel<<<numBlocks, nullptr, stream>>>(x, y, A, workSpace, tilingGm);
}
