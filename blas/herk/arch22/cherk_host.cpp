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
 * \file cherk_host.cpp
 * \brief CHERK host implementation for Atlas A2 training series (arch22).
 *
 *        C = alpha * A * A^H + beta * C   (trans = 'N')
 *        C = alpha * A^H * A + beta * C   (trans = 'C')
 *        alpha and beta are real; C is Hermitian and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Three-stage pipeline:
 *          Phase 0 (AIV) split   : complex A -> packed real Ar, Ai
 *          Phase 1 (AIC) gemm    : four real GEMMs (t1..t4)
 *          Phase 2 (AIV) combine : Cr = t1 + t2, Ci = t3 - t4, scale, write triangle
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "cherk_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

namespace {
constexpr const char* OP_TAG = "aclblasCherk";
constexpr uint32_t CHERK_TEMP_COUNT = 4;             // t1, t2, t3, t4 (trans='C' path)
constexpr uint32_t CHERK_INTERLEAVED_TEMP_COUNT = 2; // Pr, Pi (trans='N' path)
constexpr uint32_t CHERK_PACKED_COUNT = 2;           // Ar, Ai

} // namespace

// ============================================================================
//  Parameter validation
// ============================================================================
// Enum legality and leading dimensions. Must run *before* the n == 0 quick
// return: an illegal uplo/trans is an error regardless of the problem size, which
// is how reference BLAS, the shared CPU golden and the arch35 implementation all
// behave. Checking it after the quick return would make aclblasCherk report
// success for an invalid enum whenever n happens to be 0.
static aclblasStatus_t ValidateCherkEnumsAndDims(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "trans must be ACLBLAS_OP_N(111) or ACLBLAS_OP_C(113), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);

    const int ldaMin = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(lda >= ldaMin,
              OP_LOGE(OP_TAG, "lda must be >= %d for trans=%d, got %d", ldaMin, static_cast<int>(trans), lda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= max(1,n)=%d, got %d", std::max(1, n), ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Pointer checks, run only after the quick return: with n == 0 nothing is read or
// written, so null pointers are legal there.
static aclblasStatus_t ValidateCherkPointers(
    int n, int k, const float* alpha, const aclblasComplex* A, const float* beta, const aclblasComplex* C)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(A == nullptr && n > 0 && k > 0), OP_LOGE(OP_TAG, "A must not be nullptr when n>0 and k>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && n > 0), OP_LOGE(OP_TAG, "C must not be nullptr when n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Launch
// ============================================================================
// Phase 0 (split A into Ar / Ai) followed by Phase 1 (the four real GEMMs).
// Only reached when alpha and k are both non-zero, so the temps are live.
// trans='N' path: Phase 0 emits two K-interleaved operands, Phase 1 gets both
// parts of the product out of two GEMMs over the doubled K extent.
//
//   P = [ Ai_j , Ar_j ]  interleaved      Q = [ Ar_j , -Ai_j ]  interleaved
//   Pr = P*P^T = sum_j (Ai_j*Ai_j^T + Ar_j*Ar_j^T)
//   Pi = P*Q^T = sum_j (Ai_j*Ar_j^T - Ar_j*Ai_j^T)
//
// Same MAC count as the four separate GEMMs it replaces (two launches at 2k
// instead of four at k) and the same workspace (two packed operands of 2k columns
// plus two temps, against two of k columns plus four temps). What it buys is the
// accumulation order of Pi: the two signs alternate per input column, so a term
// and its counterpart cancel while the running sum is still small. Summing them
// as separate blocks lets each reach the fp32 range limit on its own, and
// Inf - Inf is NaN — which is what the RANDOM_EXTREME shapes hit.
static aclblasStatus_t RunCherkInterleavedGemms(
    void* stream, aclblasFillMode_t uplo, const CBlas3RankKShape& shape, uint32_t lda, const aclblasComplex* A,
    const CBlas3Workspace& ws, uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    CBlas3SplitConcatTilingData split{};
    split.rows = shape.aRows;
    split.cols = shape.aCols;
    split.lda = lda;
    split.packedLd = shape.aRows;
    split.blockStride = shape.aRows;    // pair partner is the next column
    split.colStride = shape.aRows * 2U; // input column j -> destination 2j, 2j+1
    const uint32_t splitBlocks = GetUsedAivCoreNum(shape.aCols, aivCoreNum);
    OP_LOGI(
        OP_TAG, "phase0 split-interleave: rows=%u cols=%u packedLd=%u aivBlocks=%u", split.rows, split.cols,
        split.packedLd, splitBlocks);
    cherk_split_interleave_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.packed[0], ws.packed[1],
        split);

    CBlas3GemmTilingData gemm = CBlas3MakeRankKGemmTiling(shape, ws.tempLdc, static_cast<uint32_t>(uplo));
    gemm.k = shape.k * 2U;          // interleaved K extent
    gemm.packedLd = split.packedLd; // unchanged: still n-long columns
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = cherk_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 gemm(interleaved): n=%u k=%u aicBlocks=%u transMode=%u kChunk=%u", gemm.m, gemm.k, gemmBlocks,
        shape.transMode, kChunk);

    uint8_t* p = ws.packed[0];
    uint8_t* q = ws.packed[1];
    CBlas3LaunchGemmChunked(cherk_gemm_kernel_do, gemmBlocks, stream, p, p, ws.temp[0], gemm, kChunk);
    CBlas3LaunchGemmChunked(cherk_gemm_kernel_do, gemmBlocks, stream, p, q, ws.temp[1], gemm, kChunk);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t RunCherkSplitAndGemms(
    void* stream, aclblasFillMode_t uplo, aclblasOperation_t trans, const CBlas3RankKShape& shape, uint32_t lda,
    const aclblasComplex* A, const CBlas3Workspace& ws, uint32_t aivCoreNum)
{
    const uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (trans == ACLBLAS_OP_N) {
        return RunCherkInterleavedGemms(stream, uplo, shape, lda, A, ws, aivCoreNum, aicCoreNum);
    }

    CBlas3SplitTilingData split{};
    split.rows = shape.aRows;
    split.cols = shape.aCols;
    split.lda = lda;
    split.packedLd = shape.aRows;
    const uint32_t splitBlocks = GetUsedAivCoreNum(shape.aCols, aivCoreNum);
    OP_LOGI(OP_TAG, "phase0 split: rows=%u cols=%u aivBlocks=%u", split.rows, split.cols, splitBlocks);
    cherk_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.packed[0], ws.packed[1],
        split);

    // Phase 1: four real GEMMs.
    //   t1 = Ar x Ar^T, t2 = Ai x Ai^T, t3 = Ai x Ar^T, t4 = Ar x Ai^T  (trans='N')
    //   t1 = Ar^T x Ar, t2 = Ai^T x Ai, t3 = Ar^T x Ai, t4 = Ai^T x Ar  (trans='C')
    //
    // t4 equals t3^T, so it could be dropped in favour of reading t3 twice, but
    // Phase 2 streams C column by column and a transposed read of t3 would
    // degenerate into per-element gathers. Recomputing t4 on the Cube is cheaper:
    // each GEMM skips the tiles outside the triangle, so the four launches
    // together cost 2*n^2*k MACs, which is exactly the arithmetic a triangular
    // complex rank-k update needs.
    const CBlas3GemmTilingData gemm = CBlas3MakeRankKGemmTiling(shape, ws.tempLdc, static_cast<uint32_t>(uplo));
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = cherk_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 gemm: n=%u k=%u aicBlocks=%u transMode=%u kChunk=%u", shape.n, shape.k, gemmBlocks,
        shape.transMode, kChunk);

    uint8_t* ar = ws.packed[0];
    uint8_t* ai = ws.packed[1];
    auto gemmInto = [&](uint8_t* l, uint8_t* r, uint8_t* out) {
        CBlas3LaunchGemmChunked(cherk_gemm_kernel_do, gemmBlocks, stream, l, r, out, gemm, kChunk);
    };
    gemmInto(ar, ar, ws.temp[0]);
    gemmInto(ai, ai, ws.temp[1]);
    if (trans == ACLBLAS_OP_N) {
        gemmInto(ai, ar, ws.temp[2]);
        gemmInto(ar, ai, ws.temp[3]);
    } else {
        gemmInto(ar, ai, ws.temp[2]);
        gemmInto(ai, ar, ws.temp[3]);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Phase 2: combine the temps, apply alpha / beta and write the uplo triangle of C.
static void RunCherkCombine(
    void* stream, aclblasFillMode_t uplo, uint32_t n, uint32_t ldc, aclblasComplex* C, const CBlas3Workspace& ws,
    bool skipTemp, bool isBetaZero, bool fusedTemps, float alphaVal, float betaVal, uint32_t aivCoreNum)
{
    CherkCombineTilingData combine{};
    combine.n = n;
    combine.ldc = ldc;
    combine.tempLdc = ws.tempLdc;
    combine.uploMode = static_cast<uint32_t>(uplo);
    combine.skipAlphaTerm = static_cast<uint32_t>(skipTemp ? 1 : 0);
    combine.isBetaZero = static_cast<uint32_t>(isBetaZero ? 1 : 0);
    combine.fusedTemps = static_cast<uint32_t>(fusedTemps ? 1 : 0);
    combine.alphaVal = alphaVal;
    combine.betaVal = betaVal;
    const uint32_t combineBlocks = GetUsedAivCoreNum(n, aivCoreNum);
    OP_LOGI(OP_TAG, "phase2 combine: n=%u aivBlocks=%u fused=%u", n, combineBlocks, combine.fusedTemps);
    if (fusedTemps) {
        // temp[0] is the whole real part and temp[1] the whole imaginary part; the
        // t2 / t4 slots are unused and get the same pointers to keep the launch
        // signature single-shaped.
        cherk_combine_kernel_do(
            combineBlocks, stream, ws.temp[0], ws.temp[0], ws.temp[1], ws.temp[1], reinterpret_cast<uint8_t*>(C),
            combine);
        return;
    }
    // t3 and t4 are deliberately swapped: Phase 2 addresses the temps with a
    // column-major stride, which transposes them, and t3^T == t4.
    cherk_combine_kernel_do(
        combineBlocks, stream, ws.temp[0], ws.temp[1], ws.temp[3], ws.temp[2], reinterpret_cast<uint8_t*>(C), combine);
}

static aclblasStatus_t LaunchCherkKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, const float* alpha,
    const aclblasComplex* A, uint32_t lda, const float* beta, aclblasComplex* C, uint32_t ldc)
{
    void* stream = h->stream;

    float alphaVal = 0.0F;
    float betaVal = 0.0F;
    const aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    const bool isBetaZero = (betaVal == 0.0F);
    const bool skipTemp = (alphaVal == 0.0F) || (k == 0U);
    OP_LOGD(
        OP_TAG, "alpha=%f beta=%f skipTemp=%d isBetaZero=%d", static_cast<double>(alphaVal),
        static_cast<double>(betaVal), static_cast<int>(skipTemp), static_cast<int>(isBetaZero));

    // Reference CHERK returns immediately in this case, leaving C - including the
    // imaginary part of its diagonal - exactly as it was.
    if (skipTemp && betaVal == 1.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const CBlas3RankKShape shape = CBlas3ResolveRankKShape(trans != ACLBLAS_OP_N, n, k);
    // The trans='N' path interleaves along K: two packed operands of twice the
    // column count, and two temps instead of four. Same total as the four-GEMM
    // form it replaces, so the workspace ceiling is unchanged.
    const bool interleaved = (trans == ACLBLAS_OP_N);
    CBlas3RankKShape packedShape = shape;
    if (interleaved) {
        packedShape.aCols = shape.aCols * 2U;
    }
    const uint32_t tempCount = interleaved ? CHERK_INTERLEAVED_TEMP_COUNT : CHERK_TEMP_COUNT;
    CBlas3Workspace ws{};
    const aclblasStatus_t wsRet =
        CBlas3PrepareWorkspace(h, packedShape, tempCount, CHERK_PACKED_COUNT, skipTemp, ws, OP_TAG);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (!skipTemp) {
        const aclblasStatus_t gemmRet = RunCherkSplitAndGemms(stream, uplo, trans, shape, lda, A, ws, aivCoreNum);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    RunCherkCombine(stream, uplo, n, ldc, C, ws, skipTemp, isBetaZero, interleaved, alphaVal, betaVal, aivCoreNum);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    const aclblasStatus_t enumSt = ValidateCherkEnumsAndDims(uplo, trans, n, k, lda, ldc);
    if (enumSt != ACLBLAS_STATUS_SUCCESS) {
        return enumSt;
    }

    // Quick return only after the enum/dimension checks; see the comment on
    // ValidateCherkEnumsAndDims.
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t ptrSt = ValidateCherkPointers(n, k, alpha, A, beta, C);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }

    return LaunchCherkKernel(
        static_cast<_aclblas_handle*>(handle), uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha,
        A, static_cast<uint32_t>(lda), beta, C, static_cast<uint32_t>(ldc));
}
