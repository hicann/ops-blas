/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_host.cpp
 * @brief Host-side implementation for Cgeam operator (complex)
 *        Handles parameter validation, tiling calculation, and kernel launch
 */

#include "cann_ops_blas.h"
#include "cgeam_kernel.h"
#include "geam_host_common.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include <algorithm>

namespace {

constexpr uint32_t UB_SIZE = 248 * 1024; // 248 KB
constexpr uint32_t UB_RESERVE = 256;
constexpr uint32_t ALIGN_UNIT = 8;              // 32 bytes / sizeof(float)
constexpr uint32_t CGEAM_BUFFERS = 12;           // 4 TQue with num=2 (A_r*2, A_i*2, C_r*2, C_i*2) + 4 TBuf (B_r, B_i, calcBufR, calcBufI)
constexpr uint32_t CGEAM_BUFFERS_BETA_ZERO = 10; // Same minus B_r, B_i (not allocated when beta=0)
// Hardware limit for DataCopyPadExt blockCount parameter.
// When curM exceeds this, DataCopyPad blockCount exceeds hardware limit
// and behavior is undefined. Each complex element maps to one blockCount
// unit (real/imag each have curM floats read separately).
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

CgeamTilingData CalCgeamTilingData(
    aclblasOperation_t transa, aclblasOperation_t transb, uint32_t m, uint32_t n, float alphaR, float alphaI,
    float betaR, float betaI, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t alphaIsZero, uint32_t betaIsZero,
    uint32_t aivCoreNum)
{
    CgeamTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.alphaR = alphaR;
    tiling.alphaI = alphaI;
    tiling.betaR = betaR;
    tiling.betaI = betaI;
    tiling.alphaIsZero = alphaIsZero;
    tiling.betaIsZero = betaIsZero;
    tiling.opA = transa;
    tiling.opB = transb;

    // Calculate tileM based on UB capacity, capped to MAX_BLOCK_COUNT
    // to avoid triggering the kernel batch-loop error path when curM > 4095.
    // For complex with beta!=0: 8 buffers (A_r, A_i, B_r, B_i, C_r, C_i, calcBufR, calcBufI)
    // For complex with beta=0: 6 buffers (A_r, A_i, C_r, C_i, calcBufR, calcBufI)
    uint32_t buffersPerTile = betaIsZero ? CGEAM_BUFFERS_BETA_ZERO : CGEAM_BUFFERS;
    uint32_t maxTileM = (UB_SIZE - UB_RESERVE) / (buffersPerTile * sizeof(float));
    uint32_t tileM = (maxTileM / ALIGN_UNIT) * ALIGN_UNIT;
    tileM = std::min(tileM, MAX_BLOCK_COUNT);
    tiling.tileM = tileM;

    // 2D block decomposition: colBlocks x mBlocks
    uint32_t colBlocks = std::min(n, aivCoreNum);
    if (colBlocks == 0) {
        colBlocks = 1;
    }
    tiling.colBlocks = colBlocks;
    tiling.perCoreN = n / colBlocks;
    tiling.remainder = n % colBlocks;

    // Calculate m-blocks
    uint32_t totalMTiles = (tiling.tileM > 0) ? ((m + tiling.tileM - 1) / tiling.tileM) : 1;
    uint32_t mBlocks = 1;
    if (colBlocks < aivCoreNum && totalMTiles > 1) {
        uint32_t maxMBlocks = aivCoreNum / colBlocks;
        mBlocks = std::min(maxMBlocks, totalMTiles);
    }
    if (mBlocks == 0) // to suppress 'maybe divide by zero' warning
        __builtin_unreachable();
    tiling.mBlocks = mBlocks;
    tiling.perCoreMTile = totalMTiles / mBlocks;
    tiling.mTileRemainder = totalMTiles % mBlocks;

    return tiling;
}

aclblasStatus_t LaunchCgeamKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* B,
    int ldb, aclblasComplex* C, int ldc)
{
    // Get core count
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgeam", "Failed to get AIV core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    float alphaR = alpha->real;
    float alphaI = alpha->imag;
    float betaR = beta->real;
    float betaI = beta->imag;
    uint32_t alphaIsZero = (alphaR == 0.0f && alphaI == 0.0f) ? 1u : 0u;
    uint32_t betaIsZero = (betaR == 0.0f && betaI == 0.0f) ? 1u : 0u;

    // Calculate tiling
    CgeamTilingData tiling = CalCgeamTilingData(
        transa, transb, static_cast<uint32_t>(m), static_cast<uint32_t>(n), alphaR, alphaI, betaR, betaI,
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), alphaIsZero, betaIsZero,
        aivCoreNum);

    // Calculate total blocks
    uint32_t numBlocks = tiling.colBlocks * tiling.mBlocks;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    OP_LOGD(
        "aclblasCgeam", "Tiling: tileM=%u, colBlocks=%u, mBlocks=%u, totalBlocks=%u", tiling.tileM, tiling.colBlocks,
        tiling.mBlocks, numBlocks);

    // Convert pointers to GM_ADDR (complex data is stored as interleaved float pairs).
    // When alpha/beta value is 0, A/B may be nullptr; use C as a dummy valid GM address
    // to avoid hardware issues with null descriptors.
    // (Kernel skips reading A/B based on alphaIsZero/betaIsZero tiling flags.)
    GM_ADDR gmC = reinterpret_cast<GM_ADDR>(C);
    GM_ADDR gmA = (A != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)) : gmC;
    GM_ADDR gmB = (B != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)) : gmC;

    // Launch kernel
    cgeam_kernel_do(gmA, gmB, gmC, tiling, numBlocks, handle->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* B,
    int ldb, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgeam", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status =
        ValidateGeamParams("aclblasCgeam", transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCgeamKernel(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
