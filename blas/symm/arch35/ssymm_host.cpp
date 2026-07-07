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
 * \file ssymm_host.cpp
 * \brief SSYMM Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ssymm_tiling_data.h"

struct SsymmMirrorTilingData;
void ssymm_mirror_kernel_do(uint8_t* gmA, uint8_t* gmWorkspaceA, const SsymmMirrorTilingData &tiling,
                             uint32_t numBlocks, void *stream);
struct SsymmGemmTilingData;
void ssymm_gemm_kernel_do(uint8_t* gmA, uint8_t* gmB, uint8_t* gmTemp,
                           const SsymmGemmTilingData &tiling, uint32_t numBlocks, void *stream);
struct SsymmScaleTilingData;
void ssymm_scale_kernel_do(uint8_t* gmTemp, uint8_t* gmC, uint8_t* gmAlpha, uint8_t* gmBeta,
                             const SsymmScaleTilingData &tiling, uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateSsymmParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    int64_t lda, int64_t ldb, int64_t ldc, const float* alpha, const float* beta,
    const float* A, const float* B, float* C)
{
    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE("aclblasSsymm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsymm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);

    int64_t dimA = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    CHECK_RET(
        lda >= dimA,
        OP_LOGE("aclblasSsymm", "lda must be >= dimA, got lda=%ld, dimA=%ld", lda, dimA);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= n,
        OP_LOGE("aclblasSsymm", "ldb must be >= n, got ldb=%ld, n=%ld", ldb, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= n,
        OP_LOGE("aclblasSsymm", "ldc must be >= n, got ldc=%ld, n=%ld", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSsymm", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSsymm", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSsymm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(B != nullptr, OP_LOGE("aclblasSsymm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(C != nullptr, OP_LOGE("aclblasSsymm", "C must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SsymmMirrorTilingData CalMirrorTilingData(
    uint32_t usedAivCoreNum, uint32_t dimA, uint32_t lda,
    uint32_t sideMode, uint32_t uploMode)
{
    SsymmMirrorTilingData tiling{};
    tiling.sideMode = sideMode;
    tiling.uploMode = uploMode;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.dimA = dimA;
    tiling.mirrorRowsPerCore = CeilDiv<uint32_t>(dimA, usedAivCoreNum);
    tiling.lda = lda;
    return tiling;
}

static SsymmGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t m, uint32_t n, uint32_t sideMode,
    uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    SsymmGemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.sideMode = sideMode;

    uint32_t coreM = CeilDiv<uint32_t>(m, usedAicCoreNum);
    uint32_t coreN = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.usedAicCoreNum = usedAicCoreNum;
    tiling.singleCoreM = std::max<uint32_t>(coreM, SSYMM_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreN, SSYMM_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(SSYMM_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(SSYMM_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = SSYMM_ARCH35_DEFAULT_TILE_K_CHUNK;

    tileM = std::max<uint32_t>(tileM, SSYMM_ARCH35_BASE_M);
    tileN = std::max<uint32_t>(tileN, SSYMM_ARCH35_BASE_N);
    tileKChunk = std::max<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_K);

    if (m < SSYMM_ARCH35_DEFAULT_TILE_M) {
        tileM = CeilAlign<uint32_t>(std::max<uint32_t>(m, SSYMM_ARCH35_BASE_M), SSYMM_ARCH35_BASE_M);
    }
    if (n < SSYMM_ARCH35_DEFAULT_TILE_N) {
        tileN = CeilAlign<uint32_t>(std::max<uint32_t>(n, SSYMM_ARCH35_BASE_N), SSYMM_ARCH35_BASE_N);
    }

    uint32_t aSideL1 = CeilAlign<uint32_t>(tileM, SSYMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_K) * SSYMM_ARCH35_FP32_SIZE;
    uint32_t bSideL1 = CeilAlign<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileN, SSYMM_ARCH35_BASE_K) * SSYMM_ARCH35_FP32_SIZE;

    while (2 * (aSideL1 + bSideL1) > SSYMM_ARCH35_L1_SIZE_BYTES && tileKChunk > SSYMM_ARCH35_BASE_K) {
        tileKChunk /= 2;
        tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_K), SSYMM_ARCH35_BASE_K);
        aSideL1 = CeilAlign<uint32_t>(tileM, SSYMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_K) * SSYMM_ARCH35_FP32_SIZE;
        bSideL1 = CeilAlign<uint32_t>(tileKChunk, SSYMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileN, SSYMM_ARCH35_BASE_K) * SSYMM_ARCH35_FP32_SIZE;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.tempRowStride = CeilAlign<uint32_t>(n, SSYMM_ARCH35_FIXPIPE_N_ALIGN);

    return tiling;
}

static SsymmScaleTilingData CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t m, uint32_t n, uint32_t ldc,
    uint32_t tempRowStride)
{
    SsymmScaleTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempRowStride = tempRowStride;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.scaleRowsPerCore = CeilDiv<uint32_t>(m, usedAivCoreNum);
    return tiling;
}

static aclblasStatus_t LaunchSsymmPipeline(
    _aclblas_handle* h, const float* A, const float* B, const float* alpha, float* C, const float* beta,
    uint32_t dimA, uint32_t uM,
    const SsymmMirrorTilingData& mirrorTiling, uint32_t usedAivCoreNum,
    const SsymmGemmTilingData& gemmTiling, uint32_t usedAicCoreNum,
    const SsymmScaleTilingData& scaleTiling, uint32_t usedAivCoreNumScale,
    uint32_t uLda)
{
    size_t workspaceASize = static_cast<size_t>(uLda) * static_cast<size_t>(dimA) * SSYMM_ARCH35_FP32_SIZE;
    size_t tempSize = static_cast<size_t>(uM) * static_cast<size_t>(gemmTiling.tempRowStride) * SSYMM_ARCH35_FP32_SIZE;
    size_t requiredBytes = workspaceASize + tempSize;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, requiredBytes);
    CHECK_RET(wsRet == ACLBLAS_STATUS_SUCCESS,
        OP_LOGE("aclblasSsymm", "workspace ensure failed, required=%zu, ret=%d", requiredBytes, wsRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    uint8_t* wsBase = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* workspaceADevice = wsBase;
    uint8_t* tempDevice = wsBase + workspaceASize;

    ssymm_mirror_kernel_do(
        (uint8_t*)A, workspaceADevice, mirrorTiling, usedAivCoreNum, h->stream);
    ssymm_gemm_kernel_do(
        workspaceADevice, (uint8_t*)B, tempDevice,
        gemmTiling, usedAicCoreNum, h->stream);
    ssymm_scale_kernel_do(
        tempDevice, (uint8_t*)C, (uint8_t*)alpha, (uint8_t*)beta,
        scaleTiling, usedAivCoreNumScale, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteSsymmKernels(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
    uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const float* A, const float* B, const float* alpha, float* C, const float* beta)
{
    uint32_t dimA = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsymm", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasSsymm", "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(dimA, aivCoreNum), 1);
    uint32_t usedAicCoreNum = std::max<uint32_t>(std::min<uint32_t>(
        CeilDiv<uint32_t>(uM, SSYMM_ARCH35_BASE_M) * CeilDiv<uint32_t>(uN, SSYMM_ARCH35_BASE_N), aicCoreNum), 1);
    uint32_t usedAivCoreNumScale = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);

    SsymmMirrorTilingData mirrorTiling = CalMirrorTilingData(
        usedAivCoreNum, dimA, uLda, static_cast<uint32_t>(side), static_cast<uint32_t>(uplo));
    SsymmGemmTilingData gemmTiling = CalGemmTilingData(
        usedAicCoreNum, uM, uN, static_cast<uint32_t>(side), uLda, uLdb, uLdc);
    SsymmScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNumScale, uM, uN, uLdc, gemmTiling.tempRowStride);

    OP_LOGD("aclblasSsymm",
        "mirror tiling: side=%u uplo=%u aivCores=%u rowsPerCore=%u lda=%u dimA=%u",
        mirrorTiling.sideMode, mirrorTiling.uploMode,
        mirrorTiling.usedAivCoreNum, mirrorTiling.mirrorRowsPerCore, mirrorTiling.lda, mirrorTiling.dimA);
    OP_LOGD("aclblasSsymm",
        "gemm tiling: m=%u n=%u side=%u aicCores=%u singleCoreM=%u singleCoreN=%u "
        "tileM=%u tileN=%u tileKChunk=%u lda=%u ldb=%u ldc=%u tempRowStride=%u",
        gemmTiling.m, gemmTiling.n, gemmTiling.sideMode, gemmTiling.usedAicCoreNum,
        gemmTiling.singleCoreM, gemmTiling.singleCoreN,
        gemmTiling.tileM, gemmTiling.tileN, gemmTiling.tileKChunk,
        gemmTiling.lda, gemmTiling.ldb, gemmTiling.ldc, gemmTiling.tempRowStride);
    OP_LOGD("aclblasSsymm",
        "scale tiling: m=%u n=%u ldc=%u tempRowStride=%u aivCores=%u rowsPerCore=%u",
        scaleTiling.m, scaleTiling.n, scaleTiling.ldc, scaleTiling.tempRowStride,
        scaleTiling.usedAivCoreNum, scaleTiling.scaleRowsPerCore);
    OP_LOGI("aclblasSsymm", "launching mirror kernel: aivCores=%u", usedAivCoreNum);
    OP_LOGI("aclblasSsymm", "launching gemm kernel: aicCores=%u", usedAicCoreNum);
    OP_LOGI("aclblasSsymm", "launching scale kernel: aivCores=%u", usedAivCoreNumScale);

    return LaunchSsymmPipeline(h, A, B, alpha, C, beta, dimA, uM,
        mirrorTiling, usedAivCoreNum, gemmTiling, usedAicCoreNum,
        scaleTiling, usedAivCoreNumScale, uLda);
}

aclblasStatus_t aclblasSsymm(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda,
    const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc)
{
    CHECK_RET(m >= 0, OP_LOGE("aclblasSsymm", "m must be >= 0, got %ld", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSsymm", "n must be >= 0, got %ld", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasSsymm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    aclblasStatus_t st = ValidateSsymmParams(side, uplo, m, n, lda, ldb, ldc, alpha, beta, A, B, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    const int64_t maxU32 = static_cast<int64_t>(UINT32_MAX);
    CHECK_RET(m <= maxU32 && n <= maxU32 && lda <= maxU32 && ldb <= maxU32 && ldc <= maxU32,
        OP_LOGE("aclblasSsymm", "dimensions exceed uint32_t limit");
        return ACLBLAS_STATUS_INVALID_VALUE);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    return ExecuteSsymmKernels(h, side, uplo,
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc),
        A, B, alpha, C, beta);
}
