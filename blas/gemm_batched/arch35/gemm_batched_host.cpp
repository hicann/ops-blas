/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_batched_host.cpp
 * \brief Batched GEMM host implementation for arch35 (DAV_3510).
 *
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
 *
 * Uses tensor_api for both cube (AIC) and vector (AIV) kernels.
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "gemm_batched_tiling_data.h"

// Kernel launchers (defined in gemm_batched_kernel.cpp)
void gemm_batched_gemm_kernel_do(uint32_t numBlocks, void* stream,
    const uint8_t* aarray, const uint8_t* barray, uint8_t* carray,
    const GemmBatchedGemmTilingData& tilingData);

void gemm_batched_alpha_beta_kernel_do(uint32_t numBlocks, void* stream,
    const uint8_t* tempAB, uint8_t* carray,
    const GemmBatchedAlphaBetaTilingData& tilingData);

void gemm_batched_early_exit_kernel_do(uint32_t numBlocks, void* stream,
    uint8_t* carray, const GemmBatchedAlphaBetaTilingData& tilingData);

// ============================================================================
// Workspace management
// ============================================================================
static aclblasStatus_t EnsureWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    return EnsureDefaultWorkspace(h, requiredSize);
}

// ============================================================================
// Validation
// ============================================================================
static aclblasStatus_t ValidateSgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, int lda, int ldb, int ldc, int batchCount,
    const float* alpha, const float* beta,
    const float* const Aarray[], const float* const Barray[], float* const Carray[])
{
    CHECK_RET(handle != nullptr,
        OP_LOGE("aclblasSgemmBatched", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr,
        OP_LOGE("aclblasSgemmBatched", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr,
        OP_LOGE("aclblasSgemmBatched", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemmBatched", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemmBatched", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasSgemmBatched", "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0,
        OP_LOGE("aclblasSgemmBatched", "batchCount must be >= 0, got %d", batchCount); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgemmBatched", "invalid transa=%d", static_cast<int>(transa)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgemmBatched", "invalid transb=%d", static_cast<int>(transb)); return ACLBLAS_STATUS_INVALID_VALUE);

    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int physRowsA = isTransA ? k : m;
    int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
        OP_LOGE("aclblasSgemmBatched", "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
        OP_LOGE("aclblasSgemmBatched", "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m),
        OP_LOGE("aclblasSgemmBatched", "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr,
            OP_LOGE("aclblasSgemmBatched", "Aarray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr,
            OP_LOGE("aclblasSgemmBatched", "Barray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr,
            OP_LOGE("aclblasSgemmBatched", "Carray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Minimum elements per AIV core to amortize launch overhead for post-process kernels.
constexpr int64_t GEMM_BATCHED_AIV_ELEMENTS_PER_CORE = 16384;

// ============================================================================
// Tiling computation
// ============================================================================
static void CalcMnBlockSplit(int m, int n, uint32_t cubeCoreNum,
    int32_t& bestMBlocks, int32_t& bestNBlocks)
{
    uint32_t baseM = GEMM_BATCHED_BASE_M;
    uint32_t baseN = GEMM_BATCHED_BASE_N;
    int32_t mTiles = (m + baseM - 1) / baseM;
    int32_t nTiles = (n + baseN - 1) / baseN;

    bestMBlocks = 1; bestNBlocks = 1;
    int32_t bestUtil = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= static_cast<int32_t>(cubeCoreNum); mb++) {
        int32_t nb = std::min(nTiles, static_cast<int32_t>(cubeCoreNum) / mb);
        if (nb < 1) nb = 1;
        int32_t util = mb * nb;
        if (util > bestUtil && util <= static_cast<int32_t>(cubeCoreNum)) {
            bestUtil = util;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    if (bestUtil == 0) { bestMBlocks = 1; bestNBlocks = 1; }
}

static aclblasStatus_t CalcGemmTiling(
    int m, int n, int k, int batchCount, int lda, int ldb, int ldc,
    bool isTransA, bool isTransB, uint32_t cubeCoreNum,
    GemmBatchedGemmTilingData& tiling)
{
    tiling = GemmBatchedGemmTilingData{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.isTransA = isTransA ? 1 : 0;
    tiling.isTransB = isTransB ? 1 : 0;
    tiling.batchCount = static_cast<uint32_t>(batchCount);
    tiling.dtypeCase = GEMM_BATCHED_DTYPE_FP32;

    tiling.tileM = GEMM_BATCHED_DEFAULT_TILE_M;
    tiling.tileN = GEMM_BATCHED_DEFAULT_TILE_N;
    tiling.tileKChunk = GEMM_BATCHED_DEFAULT_TILE_K_CHUNK;

    int32_t bestMBlocks, bestNBlocks;
    CalcMnBlockSplit(m, n, cubeCoreNum, bestMBlocks, bestNBlocks);

    uint32_t baseM = GEMM_BATCHED_BASE_M;
    uint32_t baseN = GEMM_BATCHED_BASE_N;
    tiling.mBlocks = static_cast<uint32_t>(bestMBlocks);
    tiling.nBlocks = static_cast<uint32_t>(bestNBlocks);
    tiling.singleCoreM = static_cast<uint32_t>(
        ((m + bestMBlocks - 1) / bestMBlocks + baseM - 1) / baseM * baseM);
    tiling.singleCoreN = static_cast<uint32_t>(
        ((n + bestNBlocks - 1) / bestNBlocks + baseN - 1) / baseN * baseN);
    if (tiling.singleCoreM > static_cast<uint32_t>(m)) tiling.singleCoreM = m;
    if (tiling.singleCoreN > static_cast<uint32_t>(n)) tiling.singleCoreN = n;

    int64_t totalTasks64 = static_cast<int64_t>(batchCount) * bestMBlocks * bestNBlocks;
    if (totalTasks64 > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE("aclblasSgemmBatched", "totalTasks overflow: %lld",
            static_cast<long long>(totalTasks64));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.totalTasks = static_cast<uint32_t>(totalTasks64);
    tiling.usedAicCoreNum = static_cast<uint32_t>(
        std::min(static_cast<int64_t>(cubeCoreNum), totalTasks64));

    return ACLBLAS_STATUS_SUCCESS;
}

static void ApplyColumnMajorSwap(GemmBatchedGemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
    std::swap(tiling.singleCoreM, tiling.singleCoreN);
    std::swap(tiling.mBlocks, tiling.nBlocks);
}

// ============================================================================
// Workspace helpers
// ============================================================================
static aclblasStatus_t PrepareTempWorkspace(
    _aclblas_handle* h, int batchCount, int origN, uint32_t tempRowStride,
    GemmBatchedGemmTilingData& tilingData,
    uint8_t*& workspace, size_t& alignedPtrArrayBytes)
{
    tilingData.ldc = tempRowStride;
    size_t perBatchTempBytes = static_cast<size_t>(origN) * tempRowStride * sizeof(float);
    size_t totalTempBytes = static_cast<size_t>(batchCount) * perBatchTempBytes;

    constexpr size_t TEMP_DATA_ALIGN = 64;
    size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    alignedPtrArrayBytes = (ptrArrayBytes + TEMP_DATA_ALIGN - 1) & ~(TEMP_DATA_ALIGN - 1);
    size_t totalWorkspaceBytes = alignedPtrArrayBytes + totalTempBytes;

    aclblasStatus_t wsRet = EnsureWorkspace(h, totalWorkspaceBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSgemmBatched", "workspace ensure failed, required=%zu", totalWorkspaceBytes);
        return wsRet;
    }
    workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    std::vector<uint64_t> tempPtrs(batchCount);
    uint64_t tempDataBase = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(workspace + alignedPtrArrayBytes));
    for (int i = 0; i < batchCount; i++) {
        tempPtrs[i] = tempDataBase + static_cast<uint64_t>(i) * perBatchTempBytes;
    }
    aclError aclRet = aclrtMemcpy(workspace, ptrArrayBytes, tempPtrs.data(), ptrArrayBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSgemmBatched", "Failed to copy temp ptr array, err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchAlphaBetaPostProcess(
    _aclblas_handle* h, int origM, int origN, int origLdc, uint32_t tempRowStride,
    float alphaVal, float betaVal, int batchCount, uint32_t aivCoreNum,
    uint8_t* workspace, size_t alignedPtrArrayBytes, float* const Carray[])
{
    GemmBatchedAlphaBetaTilingData abTiling{};
    abTiling.m = origM;
    abTiling.n = origN;
    abTiling.ldc = origLdc;
    abTiling.tempRowStride = static_cast<int32_t>(tempRowStride);
    abTiling.alpha = alphaVal;
    abTiling.beta = betaVal;
    abTiling.hasBeta = (betaVal != 0.0f) ? 1 : 0;
    abTiling.batchCount = batchCount;
    abTiling.dtypeCase = GEMM_BATCHED_DTYPE_FP32;
    abTiling.totalCols = static_cast<int64_t>(batchCount) * origN;

    int64_t totalElements = static_cast<int64_t>(batchCount) * origM * origN;
    abTiling.usedAivCoreNum = static_cast<int32_t>(std::min(
        static_cast<int64_t>(aivCoreNum),
        std::max(static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));

    const uint8_t* tempABData = workspace + alignedPtrArrayBytes;
    uint8_t* carrayPtrAB = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    OP_LOGI("aclblasSgemmBatched",
        "launching alpha/beta kernel: cores=%d, alpha=%.4f, beta=%.4f, batch=%d, m=%d, n=%d",
        abTiling.usedAivCoreNum, alphaVal, betaVal, batchCount, origM, origN);

    gemm_batched_alpha_beta_kernel_do(
        static_cast<uint32_t>(abTiling.usedAivCoreNum), h->stream,
        tempABData, carrayPtrAB, abTiling);
}

// ============================================================================
// Main execution
// ============================================================================
static aclblasStatus_t ValidateCoreCounts(uint32_t& cubeCoreNum, uint32_t& aivCoreNum)
{
    cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasSgemmBatched", "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemmBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteSgemmBatched(
    _aclblas_handle* h,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount)
{
    float alphaVal = *alpha;
    float betaVal = *beta;
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);

    uint32_t cubeCoreNum = 0;
    uint32_t aivCoreNum = 0;
    aclblasStatus_t coreRet = ValidateCoreCounts(cubeCoreNum, aivCoreNum);
    if (coreRet != ACLBLAS_STATUS_SUCCESS) { return coreRet; }

    GemmBatchedGemmTilingData tilingData;
    aclblasStatus_t tilingRet = CalcGemmTiling(
        m, n, k, batchCount, lda, ldb, ldc, isTransA, isTransB, cubeCoreNum, tilingData);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) {
        return tilingRet;
    }

    int origM = m, origN = n, origLdc = ldc;
    uint32_t tempRowStride = CeilAlign(static_cast<uint32_t>(origM), GEMM_BATCHED_L0C_C0);

    uint8_t* workspace = nullptr;
    size_t alignedPtrArrayBytes = 0;

    if (needPostProcess) {
        aclblasStatus_t wsRet = PrepareTempWorkspace(
            h, batchCount, origN, tempRowStride, tilingData, workspace, alignedPtrArrayBytes);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }
    }

    ApplyColumnMajorSwap(tilingData);

    OP_LOGI("aclblasSgemmBatched",
        "launching gemm kernel: cores=%u, transA=%d, transB=%d, batch=%d, m=%u, n=%u, k=%u",
        tilingData.usedAicCoreNum, isTransA, isTransB, batchCount,
        tilingData.m, tilingData.n, tilingData.k);

    const uint8_t* kernelAarray = reinterpret_cast<const uint8_t*>(Barray);
    const uint8_t* kernelBarray = reinterpret_cast<const uint8_t*>(Aarray);
    uint8_t* carrayPtr = needPostProcess
        ? workspace
        : const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    gemm_batched_gemm_kernel_do(tilingData.usedAicCoreNum, h->stream,
        kernelAarray, kernelBarray, carrayPtr, tilingData);

    if (needPostProcess) {
        LaunchAlphaBetaPostProcess(h, origM, origN, origLdc, tempRowStride,
            alphaVal, betaVal, batchCount, aivCoreNum,
            workspace, alignedPtrArrayBytes, Carray);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Early exit: k=0 or alpha=0
// ============================================================================
static aclblasStatus_t LaunchEarlyExit(
    _aclblas_handle* h, int m, int n, int ldc, int batchCount,
    float betaVal, float* const Carray[])
{
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;  // C unchanged
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemmBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    GemmBatchedAlphaBetaTilingData eeTiling{};
    eeTiling.m = m;
    eeTiling.n = n;
    eeTiling.ldc = ldc;
    eeTiling.tempRowStride = ldc;
    eeTiling.alpha = 0.0f;
    eeTiling.beta = betaVal;
    eeTiling.hasBeta = 1;
    eeTiling.batchCount = batchCount;
    eeTiling.dtypeCase = GEMM_BATCHED_DTYPE_FP32;
    eeTiling.totalCols = static_cast<int64_t>(batchCount) * n;

    int64_t totalElements = static_cast<int64_t>(batchCount) * m * n;
    eeTiling.usedAivCoreNum = static_cast<int32_t>(std::min(
        static_cast<int64_t>(aivCoreNum),
        std::max(static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));

    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    OP_LOGI("aclblasSgemmBatched",
        "early exit: beta=%.4f, batch=%d, m=%d, n=%d, cores=%d",
        betaVal, batchCount, m, n, eeTiling.usedAivCoreNum);

    gemm_batched_early_exit_kernel_do(
        static_cast<uint32_t>(eeTiling.usedAivCoreNum), h->stream,
        carrayPtr, eeTiling);

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Public API
// ============================================================================
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount)
{
    OP_LOGI("aclblasSgemmBatched",
        "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, batchCount);

    aclblasStatus_t st = ValidateSgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount,
        alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaVal = *alpha;
    float betaVal = *beta;

    // Early exit: k=0 or alpha=0
    if (k == 0 || alphaVal == 0.0f) {
        return LaunchEarlyExit(h, m, n, ldc, batchCount, betaVal, Carray);
    }

    return ExecuteSgemmBatched(h, transa, transb, m, n, k,
        alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

// ============================================================================
// Complex GEMM Batched (aclblasCgemmBatched)
//
// 4M decomposition:
//   A = Ar + j*Ai, B = Br + j*Bi
//   op(A)*op(B) = (Ar*Br - Ai*Bi) + j*(Ar*Bi + Ai*Br)
//
// Steps:
//   1. Deinterleave A → (Ar, Ai), B → (Br, Bi)  [AIV kernel]
//   2. Four real GEMM (alpha=1, beta=0, reuse fp32 cube kernel):
//        T1 = Ar*Br, T2 = Ai*Bi, T3 = Ar*Bi, T4 = Ai*Br
//   3. Combine [AIV kernel]:
//        P_r = T1 - T2, P_i = T3 + T4
//        C = alpha*(P_r + j*P_i) + beta*C_orig
// ============================================================================

// Forward declarations for complex support kernels (defined in gemm_batched_kernel.cpp)
void cgemm_batched_deinterleave_do(uint32_t numBlocks, void* stream,
    uint8_t* srcArray, uint8_t* realArray, uint8_t* imagArray,
    const CgemmBatchedDeinterleaveTilingData& tiling);

void cgemm_batched_combine_do(uint32_t numBlocks, void* stream,
    uint8_t* t1Array, uint8_t* t2Array, uint8_t* t3Array, uint8_t* t4Array,
    uint8_t* carray, const CgemmBatchedCombineTilingData& tiling);

static aclblasStatus_t ValidateCgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, int lda, int ldb, int ldc, int batchCount,
    const aclblasComplex* alpha, const aclblasComplex* beta,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[],
    aclblasComplex* const Carray[])
{
    CHECK_RET(handle != nullptr,
        OP_LOGE("aclblasCgemmBatched", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr,
        OP_LOGE("aclblasCgemmBatched", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr,
        OP_LOGE("aclblasCgemmBatched", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasCgemmBatched", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCgemmBatched", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCgemmBatched", "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0,
        OP_LOGE("aclblasCgemmBatched", "batchCount must be >= 0, got %d", batchCount); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
        OP_LOGE("aclblasCgemmBatched", "invalid transa=%d", static_cast<int>(transa)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
        OP_LOGE("aclblasCgemmBatched", "invalid transb=%d", static_cast<int>(transb)); return ACLBLAS_STATUS_INVALID_VALUE);

    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int physRowsA = isTransA ? k : m;
    int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
        OP_LOGE("aclblasCgemmBatched", "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
        OP_LOGE("aclblasCgemmBatched", "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m),
        OP_LOGE("aclblasCgemmBatched", "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr,
            OP_LOGE("aclblasCgemmBatched", "Aarray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr,
            OP_LOGE("aclblasCgemmBatched", "Barray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr,
            OP_LOGE("aclblasCgemmBatched", "Carray must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ── Complex GEMM helpers (split for R7 compliance) ──

static int32_t CalcAivCoreNum(uint32_t aivCoreNum, int64_t totalElements)
{
    return static_cast<int32_t>(std::min(static_cast<int64_t>(aivCoreNum),
        std::max(static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));
}

static aclblasStatus_t LaunchCgemmEarlyExit(
    _aclblas_handle* h, int m, int n, int ldc, int batchCount,
    float alphaR, float alphaI, float betaR, float betaI, uint32_t aivCoreNum,
    aclblasComplex* const Carray[])
{
    uint32_t tempRowStride = CeilAlign(static_cast<uint32_t>(m), GEMM_BATCHED_L0C_C0);
    size_t perBatchTempBytes = static_cast<size_t>(n) * tempRowStride * sizeof(float);
    size_t totalTempBytes = static_cast<size_t>(batchCount) * perBatchTempBytes;
    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    constexpr size_t ALIGN64 = 64;
    size_t alignedPtrBytes = (ptrBytes + ALIGN64 - 1) & ~(ALIGN64 - 1);
    // Early-exit: T1=T2=T3=T4 all zero. Allocate a single pointer array and
    // pass it four times to the combine kernel instead of 4 identical copies.
    size_t totalWorkspace = alignedPtrBytes + totalTempBytes;
    aclblasStatus_t wsRet = EnsureWorkspace(h, totalWorkspace);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) { return wsRet; }
    uint8_t* ws = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    aclError aclRet = aclrtMemset(ws + alignedPtrBytes, totalTempBytes, 0, totalTempBytes);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "aclrtMemset failed err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    std::vector<uint64_t> ptrs(batchCount);
    uint64_t base = reinterpret_cast<uint64_t>(ws + alignedPtrBytes);
    for (int i = 0; i < batchCount; i++) {
        ptrs[i] = base + static_cast<uint64_t>(i) * perBatchTempBytes;
    }
    aclRet = aclrtMemcpy(ws, ptrBytes, ptrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "aclrtMemcpy failed err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmBatchedCombineTilingData cTiling{};
    cTiling.m = m; cTiling.n = n; cTiling.ldc = ldc;
    cTiling.tempRowStride = static_cast<int32_t>(tempRowStride);
    cTiling.alphaReal = alphaR; cTiling.alphaImag = alphaI;
    cTiling.betaReal = betaR; cTiling.betaImag = betaI;
    cTiling.hasBeta = (betaR != 0.0f || betaI != 0.0f) ? 1 : 0;
    cTiling.batchCount = batchCount;
    cTiling.totalCols = static_cast<int64_t>(batchCount) * n;
    cTiling.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, static_cast<int64_t>(batchCount) * m * n);
    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    cgemm_batched_combine_do(static_cast<uint32_t>(cTiling.usedAivCoreNum), h->stream,
        ws, ws, ws, ws, carrayPtr, cTiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static CgemmBatchedDeinterleaveTilingData BuildDeinterleaveTiling(
    int physRows, int physCols, int ld, int batchCount, bool isConj, uint32_t aivCoreNum)
{
    CgemmBatchedDeinterleaveTilingData t{};
    t.m = physRows;
    t.k = physCols;
    t.lda = ld;
    t.batchCount = batchCount;
    t.isConjugate = isConj ? 1 : 0;
    int64_t elems = static_cast<int64_t>(batchCount) * physRows * physCols;
    t.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, elems);
    return t;
}

struct CgemmBatchedWsCtx {
    uint8_t* ws;
    size_t ptrRegion;
    size_t alignedPtrBytes;
    uint32_t tempRowStride;
    int physRowsA, physColsA, physRowsB, physColsB;
    size_t aPerBatch, bPerBatch, tPerBatch;
    size_t aTotal, bTotal, tTotal;
    size_t arOff, aiOff, brOff, biOff, t1Off, t2Off, t3Off, t4Off;
};

static aclblasStatus_t CopyCgemmPtrArrays(uint8_t* ws, const CgemmBatchedWsCtx& ctx, int batchCount)
{
    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    std::vector<uint64_t> arPtrs(batchCount), aiPtrs(batchCount), brPtrs(batchCount), biPtrs(batchCount);
    std::vector<uint64_t> t1Ptrs(batchCount), t2Ptrs(batchCount), t3Ptrs(batchCount), t4Ptrs(batchCount);
    for (int i = 0; i < batchCount; i++) {
        arPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.arOff + i * ctx.aPerBatch);
        aiPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.aiOff + i * ctx.aPerBatch);
        brPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.brOff + i * ctx.bPerBatch);
        biPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.biOff + i * ctx.bPerBatch);
        t1Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t1Off + i * ctx.tPerBatch);
        t2Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t2Off + i * ctx.tPerBatch);
        t3Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t3Off + i * ctx.tPerBatch);
        t4Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t4Off + i * ctx.tPerBatch);
    }
    auto ptrArrAt = [&](int idx) -> uint8_t* { return ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    auto copyPtrArr = [&](uint8_t* dst, const std::vector<uint64_t>& src) -> bool {
        return aclrtMemcpy(dst, ptrBytes, src.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    };
    if (!copyPtrArr(ptrArrAt(0), arPtrs) || !copyPtrArr(ptrArrAt(1), aiPtrs) ||
        !copyPtrArr(ptrArrAt(2), brPtrs) || !copyPtrArr(ptrArrAt(3), biPtrs) ||
        !copyPtrArr(ptrArrAt(4), t1Ptrs) || !copyPtrArr(ptrArrAt(5), t2Ptrs) ||
        !copyPtrArr(ptrArrAt(6), t3Ptrs) || !copyPtrArr(ptrArrAt(7), t4Ptrs)) {
        OP_LOGE("aclblasCgemmBatched", "failed to copy pointer arrays");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t PrepareCgemmWorkspace(
    _aclblas_handle* h, int m, int n, int k, int lda, int ldb, int batchCount,
    bool isTransA, bool isTransB, CgemmBatchedWsCtx& ctx)
{
    int physRowsA = isTransA ? k : m;
    int physColsA = isTransA ? m : k;
    int physRowsB = isTransB ? n : k;
    int physColsB = isTransB ? k : n;

    ctx.tempRowStride = CeilAlign(static_cast<uint32_t>(m), GEMM_BATCHED_L0C_C0);
    ctx.aPerBatch = static_cast<size_t>(std::max(1, lda)) * std::max(1, physColsA) * sizeof(float);
    ctx.bPerBatch = static_cast<size_t>(std::max(1, ldb)) * std::max(1, physColsB) * sizeof(float);
    ctx.tPerBatch = static_cast<size_t>(n) * ctx.tempRowStride * sizeof(float);
    ctx.physRowsA = physRowsA; ctx.physColsA = physColsA;
    ctx.physRowsB = physRowsB; ctx.physColsB = physColsB;

    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    constexpr size_t ALIGN64 = 64;
    ctx.alignedPtrBytes = (ptrBytes + ALIGN64 - 1) & ~(ALIGN64 - 1);
    ctx.aTotal = ctx.aPerBatch * batchCount;
    ctx.bTotal = ctx.bPerBatch * batchCount;
    ctx.tTotal = ctx.tPerBatch * batchCount;
    ctx.ptrRegion = ctx.alignedPtrBytes * 8;
    size_t dataRegion = ctx.aTotal * 2 + ctx.bTotal * 2 + ctx.tTotal * 4;
    size_t totalWorkspace = ctx.ptrRegion + dataRegion;

    aclblasStatus_t wsRet = EnsureWorkspace(h, totalWorkspace);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "workspace ensure failed, required=%zu", totalWorkspace);
        return wsRet;
    }
    ctx.ws = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    ctx.arOff = 0;
    ctx.aiOff = ctx.arOff + ctx.aTotal;
    ctx.brOff = ctx.aiOff + ctx.aTotal;
    ctx.biOff = ctx.brOff + ctx.bTotal;
    ctx.t1Off = ctx.biOff + ctx.bTotal;
    ctx.t2Off = ctx.t1Off + ctx.tTotal;
    ctx.t3Off = ctx.t2Off + ctx.tTotal;
    ctx.t4Off = ctx.t3Off + ctx.tTotal;

    return CopyCgemmPtrArrays(ctx.ws, ctx, batchCount);
}

static aclblasStatus_t LaunchCgemmDeinterleaveAndGemm(
    _aclblas_handle* h, int m, int n, int k, int lda, int ldb, int batchCount,
    aclblasOperation_t transa, aclblasOperation_t transb,
    uint32_t cubeCoreNum, uint32_t aivCoreNum,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[],
    const CgemmBatchedWsCtx& ctx)
{
    auto ptrArrAt = [&](int idx) -> uint8_t* { return ctx.ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    uint8_t* aPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Aarray));
    uint8_t* bPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Barray));

    bool isConjA = (transa == ACLBLAS_OP_C);
    bool isConjB = (transb == ACLBLAS_OP_C);
    aclblasOperation_t realTransA = isConjA ? ACLBLAS_OP_T : transa;
    aclblasOperation_t realTransB = isConjB ? ACLBLAS_OP_T : transb;

    auto dTilingA = BuildDeinterleaveTiling(ctx.physRowsA, ctx.physColsA, lda, batchCount, isConjA, aivCoreNum);
    auto dTilingB = BuildDeinterleaveTiling(ctx.physRowsB, ctx.physColsB, ldb, batchCount, isConjB, aivCoreNum);

    OP_LOGI("aclblasCgemmBatched", "deinterleave: A cores=%d, B cores=%d, conjA=%d, conjB=%d",
        dTilingA.usedAivCoreNum, dTilingB.usedAivCoreNum, isConjA, isConjB);

    cgemm_batched_deinterleave_do(static_cast<uint32_t>(dTilingA.usedAivCoreNum), h->stream,
        aPtrBuf, ptrArrAt(0), ptrArrAt(1), dTilingA);
    cgemm_batched_deinterleave_do(static_cast<uint32_t>(dTilingB.usedAivCoreNum), h->stream,
        bPtrBuf, ptrArrAt(2), ptrArrAt(3), dTilingB);

    const float* const* arArr = reinterpret_cast<const float* const*>(ptrArrAt(0));
    const float* const* aiArr = reinterpret_cast<const float* const*>(ptrArrAt(1));
    const float* const* brArr = reinterpret_cast<const float* const*>(ptrArrAt(2));
    const float* const* biArr = reinterpret_cast<const float* const*>(ptrArrAt(3));
    float* const* t1Arr = reinterpret_cast<float* const*>(ptrArrAt(4));
    float* const* t2Arr = reinterpret_cast<float* const*>(ptrArrAt(5));
    float* const* t3Arr = reinterpret_cast<float* const*>(ptrArrAt(6));
    float* const* t4Arr = reinterpret_cast<float* const*>(ptrArrAt(7));

    bool isRealTransA = (realTransA != ACLBLAS_OP_N);
    bool isRealTransB = (realTransB != ACLBLAS_OP_N);
    int ldcTemp = static_cast<int>(ctx.tempRowStride);
    GemmBatchedGemmTilingData realGemmTiling;
    aclblasStatus_t tilingRet = CalcGemmTiling(m, n, k, batchCount, lda, ldb, ldcTemp,
        isRealTransA, isRealTransB, cubeCoreNum, realGemmTiling);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) { return tilingRet; }
    ApplyColumnMajorSwap(realGemmTiling);

    auto launchRealGemm = [&](const float* const* aArr, const float* const* bArr, float* const* cArr) {
        const uint8_t* kernelA = reinterpret_cast<const uint8_t*>(bArr);
        const uint8_t* kernelB = reinterpret_cast<const uint8_t*>(aArr);
        uint8_t* cPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(cArr));
        gemm_batched_gemm_kernel_do(realGemmTiling.usedAicCoreNum, h->stream,
            kernelA, kernelB, cPtr, realGemmTiling);
    };
    launchRealGemm(arArr, brArr, t1Arr);
    launchRealGemm(aiArr, biArr, t2Arr);
    launchRealGemm(arArr, biArr, t3Arr);
    launchRealGemm(aiArr, brArr, t4Arr);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCgemmMainPath(
    _aclblas_handle* h,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, int lda, int ldb, int ldc, int batchCount,
    float alphaR, float alphaI, float betaR, float betaI,
    uint32_t cubeCoreNum, uint32_t aivCoreNum,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[],
    aclblasComplex* const Carray[])
{
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);

    CgemmBatchedWsCtx ctx{};
    aclblasStatus_t prepRet = PrepareCgemmWorkspace(
        h, m, n, k, lda, ldb, batchCount, isTransA, isTransB, ctx);
    if (prepRet != ACLBLAS_STATUS_SUCCESS) { return prepRet; }

    aclblasStatus_t gemmRet = LaunchCgemmDeinterleaveAndGemm(
        h, m, n, k, lda, ldb, batchCount, transa, transb,
        cubeCoreNum, aivCoreNum, Aarray, Barray, ctx);
    if (gemmRet != ACLBLAS_STATUS_SUCCESS) { return gemmRet; }

    auto ptrArrAt = [&](int idx) -> uint8_t* { return ctx.ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    uint8_t* cPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    CgemmBatchedCombineTilingData cTiling{};
    cTiling.m = m; cTiling.n = n; cTiling.ldc = ldc;
    cTiling.tempRowStride = static_cast<int32_t>(ctx.tempRowStride);
    cTiling.alphaReal = alphaR; cTiling.alphaImag = alphaI;
    cTiling.betaReal = betaR; cTiling.betaImag = betaI;
    cTiling.hasBeta = (betaR != 0.0f || betaI != 0.0f) ? 1 : 0;
    cTiling.batchCount = batchCount;
    cTiling.totalCols = static_cast<int64_t>(batchCount) * n;
    cTiling.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, static_cast<int64_t>(batchCount) * m * n);

    OP_LOGI("aclblasCgemmBatched", "combine: cores=%d, alpha=(%f,%f), beta=(%f,%f)",
        cTiling.usedAivCoreNum, alphaR, alphaI, betaR, betaI);

    cgemm_batched_combine_do(static_cast<uint32_t>(cTiling.usedAivCoreNum), h->stream,
        ptrArrAt(4), ptrArrAt(5), ptrArrAt(6), ptrArrAt(7), cPtrBuf, cTiling);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const Barray[], int ldb,
    const aclblasComplex* beta,
    aclblasComplex* const Carray[], int ldc,
    int batchCount)
{
    OP_LOGI("aclblasCgemmBatched",
        "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, batchCount);

    aclblasStatus_t st = ValidateCgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount,
        alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaR = alpha->real;
    float alphaI = alpha->imag;
    float betaR = beta->real;
    float betaI = beta->imag;

    if (k == 0 || (alphaR == 0.0f && alphaI == 0.0f)) {
        if (betaR == 1.0f && betaI == 0.0f) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        uint32_t aivCoreNum = GetAivCoreCount();
        if (aivCoreNum == 0) {
            OP_LOGE("aclblasCgemmBatched", "GetAivCoreCount failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return LaunchCgemmEarlyExit(h, m, n, ldc, batchCount, alphaR, alphaI, betaR, betaI, aivCoreNum, Carray);
    }

    uint32_t cubeCoreNum = 0;
    uint32_t aivCoreNum = 0;
    aclblasStatus_t coreRet = ValidateCoreCounts(cubeCoreNum, aivCoreNum);
    if (coreRet != ACLBLAS_STATUS_SUCCESS) { return coreRet; }

    return LaunchCgemmMainPath(h, transa, transb, m, n, k, lda, ldb, ldc, batchCount,
        alphaR, alphaI, betaR, betaI, cubeCoreNum, aivCoreNum, Aarray, Barray, Carray);
}
