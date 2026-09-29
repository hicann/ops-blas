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
 * \file cher2k_host.cpp
 * \brief CHER2K host implementation for Atlas A2 training series (arch22).
 *
 *        C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C   (trans = 'N', A/B are n x k)
 *        C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C   (trans = 'C', A/B are k x n)
 *        alpha is complex, beta is real; C is Hermitian and only the uplo triangle
 *        is referenced and updated. All matrices are column-major complex64.
 *
 *        Pipeline:
 *          Phase 0 (AIV) split   : A -> Ar, Ai and B -> Br, Bi   (two launches)
 *          Phase 1 (AIC) gemm    : eight GEMMs producing both orientations of M
 *          Phase 2 (AIV) combine : Hermitian assembly and scaling
 *
 *        With M = A*B^H, B*A^H = M^H, so the operator reduces to
 *          C = alpha*M + conj(alpha)*M^H + beta*C
 *        whose expansion needs Mr, Mr^T, Mi and Mi^T. A and B are distinct, so no
 *        orientation is free (in CHERK the fourth product is the transpose of the
 *        third, which is why it needs only four GEMMs).
 *
 *        Why not the K-concatenation trick used by CSYRK: it would cut the launches
 *        from eight to four, but needs four 2kn packed buffers, which at n=k=8000
 *        totals 2.87GiB against the library's 2GiB workspace cap. The plain layout
 *        fits in 1.92GiB. CHER2K also has no catastrophic cancellation to fix --
 *        both terms of Mi = Ai*Br^T - Ar*Bi^T are already O(sqrt(k)) cross
 *        products, unlike CSYRK's real part where two O(k) positive-definite sums
 *        cancel down to O(sqrt(k)).
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "cher2k_kernel.h"

namespace {
constexpr const char* OP_TAG = "aclblasCher2k";
constexpr uint32_t CHER2K_TEMP_COUNT = 4;   // Mr, Mr^T, Mi, Mi^T
constexpr uint32_t CHER2K_PACKED_COUNT = 4; // Ar, Ai, Br, Bi

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }
} // namespace

// ============================================================================
//  Parameter validation
// ============================================================================
// Enum legality and leading dimensions, checked before the n == 0 quick return.
static aclblasStatus_t ValidateCher2kEnumsAndDims(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    // CHER2K forms A*B^H, so the conjugate transpose is the meaningful option; a
    // plain transpose would break Hermitian symmetry and is rejected.
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "trans must be ACLBLAS_OP_N(111) or ACLBLAS_OP_C(113), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    const int ldMin = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(lda >= ldMin,
              OP_LOGE(OP_TAG, "lda must be >= %d for trans=%d, got %d", ldMin, static_cast<int>(trans), lda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= ldMin,
              OP_LOGE(OP_TAG, "ldb must be >= %d for trans=%d, got %d", ldMin, static_cast<int>(trans), ldb);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= max(1,n)=%d, got %d", std::max(1, n), ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCher2kPointers(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B, const float* beta,
    const aclblasComplex* C)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(A == nullptr && n > 0 && k > 0), OP_LOGE(OP_TAG, "A must not be nullptr when n>0 and k>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(B == nullptr && n > 0 && k > 0), OP_LOGE(OP_TAG, "B must not be nullptr when n>0 and k>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && n > 0), OP_LOGE(OP_TAG, "C must not be nullptr when n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Launch
// ============================================================================
// alpha is complex and beta a single real, so the two cannot share one read.
static aclblasStatus_t ReadCher2kScalars(
    void* stream, const aclblasComplex* alpha, const float* beta, aclblasComplex& alphaVal, float& betaVal)
{
    aclblasComplex betaAsComplex{0.0F, 0.0F};
    aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, alpha, alphaVal, betaAsComplex, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }
    float betaDummy = 0.0F;
    return CBlas3ReadScalarsFromDevice(beta, beta, betaVal, betaDummy, stream, OP_TAG);
}

// Phase 0: split A and B. Both have the same logical shape, so one tiling
// description serves both launches.
static void RunCher2kSplit(
    void* stream, const CBlas3RankKShape& shape, uint32_t lda, uint32_t ldb, const aclblasComplex* A,
    const aclblasComplex* B, const CBlas3Workspace& ws, uint32_t aivCoreNum)
{
    CBlas3SplitTilingData split{};
    split.rows = shape.aRows;
    split.cols = shape.aCols;
    split.lda = lda;
    split.packedLd = shape.aRows;
    const uint32_t splitBlocks = std::min(aivCoreNum, std::max(1U, shape.aCols));
    OP_LOGI(OP_TAG, "phase0 split: rows=%u cols=%u aivBlocks=%u", split.rows, split.cols, splitBlocks);
    cher2k_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.packed[0], ws.packed[1],
        split);
    split.lda = ldb;
    cher2k_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)), ws.packed[2], ws.packed[3],
        split);
}

// Phase 1: eight GEMMs, two per temp.
//
//   Mr   = Ar*Br^T + Ai*Bi^T        Mr^T = Br*Ar^T + Bi*Ai^T
//   Mi   = Ai*Br^T - Ar*Bi^T        Mi^T = Br*Ai^T - Bi*Ar^T
//
// For trans='C' the operator forms A^H*B instead of A*B^H, which gives
//   Mr = Ar^T*Br + Ai^T*Bi        Mi = Ar^T*Bi - Ai^T*Br
// The eight products are the *same* operand pairs as for trans='N'; only the
// imaginary part's two signs are exchanged, so the launch sequence is shared and
// only the buffer that gets negated differs.
//
// Atomic accumulation can only add, so the two subtractions are obtained by
// flipping the sign of one packed buffer in place once the real part is done with
// it. That buffer appears positively exactly twice and negatively exactly twice,
// so a single flip suffices and no fifth packed buffer is needed (which at
// n=k=8000 would not fit).
static aclblasStatus_t RunCher2kGemms(
    void* stream, aclblasFillMode_t uplo, aclblasOperation_t trans, const CBlas3RankKShape& shape,
    const CBlas3Workspace& ws, uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    // Checked before anything is queued: the negate kernel counts elements in a
    // uint32_t, and truncating would flip only the first 2^32 floats while the
    // GEMMs still accumulate over the whole K -- a wrong sign rather than a
    // visible failure. Bailing out here keeps the failure free of side effects.
    // A buffer this large is already past the workspace cap, which is why the
    // check reports the same status.
    const uint64_t negCount = static_cast<uint64_t>(shape.aRows) * shape.aCols;
    if (negCount > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(
            OP_TAG, "packed plane has %llu floats, beyond the negate kernel's 32-bit element count",
            static_cast<unsigned long long>(negCount));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    CBlas3GemmTilingData gemm = CBlas3MakeRankKGemmTiling(shape, ws.tempLdc, static_cast<uint32_t>(uplo));
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = cher2k_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 gemm: n=%u k=%u aicBlocks=%u transMode=%u kChunk=%u", shape.n, shape.k, gemmBlocks,
        shape.transMode, kChunk);

    uint8_t* ar = ws.packed[0];
    uint8_t* ai = ws.packed[1];
    uint8_t* br = ws.packed[2];
    uint8_t* bi = ws.packed[3];

    auto gemmInto = [&](uint8_t* l, uint8_t* r, uint8_t* out, bool accumulate) {
        CBlas3LaunchGemmChunked(cher2k_gemm_kernel_do, gemmBlocks, stream, l, r, out, gemm, kChunk, accumulate);
    };

    gemmInto(ar, br, ws.temp[0], false); // Mr   = Ar*Br^T   (N) / Ar^T*Br (C)
    gemmInto(ai, bi, ws.temp[0], true);  //        + Ai*Bi^T  (N) / Ai^T*Bi (C)
    gemmInto(br, ar, ws.temp[1], false); // Mr^T = Br*Ar^T   (N) / Br^T*Ar (C)
    gemmInto(bi, ai, ws.temp[1], true);  //        + Bi*Ai^T  (N) / Bi^T*Ai (C)

    // Taken from trans, not from shape.transMode: the packed buffers are stored
    // column-major, so Matmul already sees them transposed and the resolver maps
    // trans='N' to TRANS_LEFT. Reading the sign choice off transMode would invert
    // it. Same stream, so the flip is ordered between the two groups of four.
    const bool isConjTrans = (trans != ACLBLAS_OP_N);
    uint8_t* negBuf = isConjTrans ? br : bi;
    OP_LOGI(
        OP_TAG, "phase1 negate %s in place: %llu floats", isConjTrans ? "Br" : "Bi",
        static_cast<unsigned long long>(negCount));
    cher2k_negate_kernel_do(aivCoreNum, stream, negBuf, static_cast<uint32_t>(negCount));

    // With negBuf flipped, the positive product of each pair is listed first.
    if (isConjTrans) {
        gemmInto(ar, bi, ws.temp[2], false); // Mi   =  Ar^T*Bi
        gemmInto(ai, br, ws.temp[2], true);  //        - Ai^T*Br
        gemmInto(bi, ar, ws.temp[3], false); // Mi^T =  Bi^T*Ar
        gemmInto(br, ai, ws.temp[3], true);  //        - Br^T*Ai
    } else {
        gemmInto(ai, br, ws.temp[2], false); // Mi   =  Ai*Br^T
        gemmInto(ar, bi, ws.temp[2], true);  //        - Ar*Bi^T
        gemmInto(br, ai, ws.temp[3], false); // Mi^T =  Br*Ai^T
        gemmInto(bi, ar, ws.temp[3], true);  //        - Bi*Ar^T
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Phase 2: Hermitian assembly.
//
// Phase 2 reads the temps with a column-major stride, which transposes them, so
// the buffer holding Mr^T is the one whose read yields Mr. The pairs are
// therefore passed swapped. Getting this backwards flips the sign of the
// imaginary part, which is antisymmetric.
static void RunCher2kCombine(
    void* stream, aclblasFillMode_t uplo, uint32_t n, uint32_t ldc, aclblasComplex* C, const CBlas3Workspace& ws,
    bool skipTemp, bool isBetaZero, const aclblasComplex& alphaVal, float betaVal, uint32_t aivCoreNum)
{
    Cher2kCombineTilingData combine{};
    combine.n = n;
    combine.ldc = ldc;
    combine.tempLdc = ws.tempLdc;
    combine.uploMode = static_cast<uint32_t>(uplo);
    combine.skipAlphaTerm = static_cast<uint32_t>(skipTemp ? 1 : 0);
    combine.isBetaZero = static_cast<uint32_t>(isBetaZero ? 1 : 0);
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaVal = betaVal;
    const uint32_t combineBlocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase2 combine: n=%u aivBlocks=%u", n, combineBlocks);
    cher2k_combine_kernel_do(
        combineBlocks, stream, ws.temp[1], ws.temp[0], ws.temp[3], ws.temp[2], reinterpret_cast<uint8_t*>(C), combine);
}

static aclblasStatus_t LaunchCher2kKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    const float* beta, aclblasComplex* C, uint32_t ldc)
{
    void* stream = h->stream;

    aclblasComplex alphaVal{0.0F, 0.0F};
    float betaVal = 0.0F;
    const aclblasStatus_t readRet = ReadCher2kScalars(stream, alpha, beta, alphaVal, betaVal);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    const bool skipTemp = IsComplexZero(alphaVal) || (k == 0U);
    const bool isBetaZero = (betaVal == 0.0F);
    OP_LOGD(
        OP_TAG, "alpha=(%f,%f) beta=%f skipTemp=%d isBetaZero=%d", static_cast<double>(alphaVal.real),
        static_cast<double>(alphaVal.imag), static_cast<double>(betaVal), static_cast<int>(skipTemp),
        static_cast<int>(isBetaZero));

    if (skipTemp && betaVal == 1.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const CBlas3RankKShape shape = CBlas3ResolveRankKShape(trans != ACLBLAS_OP_N, n, k);
    CBlas3Workspace ws{};
    const aclblasStatus_t wsRet =
        CBlas3PrepareWorkspace(h, shape, CHER2K_TEMP_COUNT, CHER2K_PACKED_COUNT, skipTemp, ws, OP_TAG);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (!skipTemp) {
        const uint32_t aicCoreNum = GetAicCoreCount();
        if (aicCoreNum == 0U) {
            OP_LOGE(OP_TAG, "GetAicCoreCount returned 0");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        RunCher2kSplit(stream, shape, lda, ldb, A, B, ws, aivCoreNum);
        const aclblasStatus_t gemmRet = RunCher2kGemms(stream, uplo, trans, shape, ws, aivCoreNum, aicCoreNum);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    RunCher2kCombine(stream, uplo, n, ldc, C, ws, skipTemp, isBetaZero, alphaVal, betaVal, aivCoreNum);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t enumSt = ValidateCher2kEnumsAndDims(uplo, trans, n, k, lda, ldb, ldc);
    if (enumSt != ACLBLAS_STATUS_SUCCESS) {
        return enumSt;
    }

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t ptrSt = ValidateCher2kPointers(n, k, alpha, A, B, beta, C);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }

    return LaunchCher2kKernel(
        static_cast<_aclblas_handle*>(handle), uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha,
        A, static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), beta, C, static_cast<uint32_t>(ldc));
}
