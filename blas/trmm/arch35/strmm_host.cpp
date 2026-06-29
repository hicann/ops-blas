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
 * \file strmm_host.cpp
 * \brief STRMM Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "strmm_tiling_data.h"

static aclblasStatus_t EnsureWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    if (h == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    size_t availableSize = aclblasGetEffectiveWorkspaceSize(h);
    if (requiredSize <= availableSize) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (h->use_user_workspace) {
        OP_LOGE("aclblasStrmm", "user workspace too small: required=%zu, available=%zu",
                 requiredSize, availableSize);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (h->stream != nullptr) {
        aclError aclRet = aclrtSynchronizeStream(h->stream);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    size_t newSize = std::max(requiredSize, h->default_workspace_size * 2);
    if (h->default_workspace != nullptr) {
        aclError freeRet = aclrtFree(h->default_workspace);
        if (freeRet != ACL_SUCCESS) {
            OP_LOGE("aclblasStrmm", "aclrtFree old workspace failed, ret=%d", freeRet);
        }
        h->default_workspace = nullptr;
        h->default_workspace_size = 0;
    }
    void* ptr = nullptr;
    aclError aclRet = aclrtMalloc(&ptr, newSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    h->default_workspace = ptr;
    h->default_workspace_size = newSize;
    return ACLBLAS_STATUS_SUCCESS;
}

struct StrmmMirrorTilingData;
void strmm_mirror_kernel_do(const uint8_t* gmA, uint8_t* gmWorkspaceA, const StrmmMirrorTilingData &tiling,
                              uint32_t numBlocks, void *stream);
struct StrmmGemmTilingData;
void strmm_gemm_kernel_do(const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp,
                            const StrmmGemmTilingData &tiling, uint32_t numBlocks, void *stream);
struct StrmmScaleTilingData;
void strmm_scale_kernel_do(uint8_t* gmTemp, uint8_t* gmB, float alpha,
                              const StrmmScaleTilingData &tiling, uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateStrmmParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transA,
    aclblasDiagType_t diag, int64_t m, int64_t n,
    int64_t lda, int64_t ldb, const float* alpha, const float* A, float* B)
{
    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE("aclblasStrmm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasStrmm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        transA == ACLBLAS_OP_N || transA == ACLBLAS_OP_T || transA == ACLBLAS_OP_C,
        OP_LOGE("aclblasStrmm", "transA must be N, T or C, got %d", static_cast<int>(transA));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
        OP_LOGE("aclblasStrmm", "diag must be UNIT or NON_UNIT, got %d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE);

    int64_t dimA = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    CHECK_RET(
        lda >= dimA,
        OP_LOGE("aclblasStrmm", "lda must be >= dimA, got lda=%ld, dimA=%ld", lda, dimA);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= n,
        OP_LOGE("aclblasStrmm", "ldb must be >= n, got ldb=%ld, n=%ld", ldb, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasStrmm", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasStrmm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(B != nullptr, OP_LOGE("aclblasStrmm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static StrmmMirrorTilingData CalMirrorTilingData(
    uint32_t usedAivCoreNum, uint32_t m, uint32_t n, uint32_t lda,
    uint32_t sideMode, uint32_t uploMode, uint32_t transMode, uint32_t diagMode)
{
    StrmmMirrorTilingData tiling{};
    tiling.sideMode = sideMode;
    tiling.uploMode = uploMode;
    tiling.transMode = transMode;
    tiling.diagMode = diagMode;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.dimA = (sideMode == ACLBLAS_SIDE_LEFT) ? m : n;
    tiling.mirrorRowsPerCore = CeilDiv<uint32_t>(tiling.dimA, usedAivCoreNum);
    tiling.lda = lda;
    return tiling;
}

static StrmmGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t m, uint32_t n, uint32_t sideMode,
    uint32_t lda, uint32_t ldb)
{
    StrmmGemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.sideMode = sideMode;

    uint32_t coreM = CeilDiv<uint32_t>(m, usedAicCoreNum);
    uint32_t coreN = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.usedAicCoreNum = usedAicCoreNum;
    tiling.singleCoreM = std::max<uint32_t>(coreM, STRMM_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreN, STRMM_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(STRMM_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(STRMM_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = STRMM_ARCH35_DEFAULT_TILE_K_CHUNK;

    tileM = std::max<uint32_t>(tileM, STRMM_ARCH35_BASE_M);
    tileN = std::max<uint32_t>(tileN, STRMM_ARCH35_BASE_N);
    tileKChunk = std::max<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K);

    if (m < STRMM_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(std::max<uint32_t>(m, STRMM_ARCH35_BASE_M), STRMM_ARCH35_BASE_M));
    }
    if (n < STRMM_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(std::max<uint32_t>(n, STRMM_ARCH35_BASE_N), STRMM_ARCH35_BASE_N));
    }

    uint32_t aSideL1 = CeilAlign<uint32_t>(tileM, STRMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * STRMM_ARCH35_FP32_SIZE;
    uint32_t bSideL1 = CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * CeilAlign<uint32_t>(tileN, STRMM_ARCH35_BASE_M) * STRMM_ARCH35_FP32_SIZE;

    while (2 * (aSideL1 + bSideL1) > STRMM_ARCH35_L1_SIZE_BYTES && tileKChunk > STRMM_ARCH35_BASE_K) {
        tileKChunk /= 2;
        tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K), STRMM_ARCH35_BASE_K);
        aSideL1 = CeilAlign<uint32_t>(tileM, STRMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * STRMM_ARCH35_FP32_SIZE;
        bSideL1 = CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * CeilAlign<uint32_t>(tileN, STRMM_ARCH35_BASE_M) * STRMM_ARCH35_FP32_SIZE;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.tempRowStride = CeilAlign<uint32_t>(n, STRMM_ARCH35_FIXPIPE_N_ALIGN);

    return tiling;
}

static StrmmScaleTilingData CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t m, uint32_t n, uint32_t ldb,
    uint32_t tempRowStride)
{
    StrmmScaleTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldb = ldb;
    tiling.tempRowStride = tempRowStride;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.scaleRowsPerCore = CeilDiv<uint32_t>(m, usedAivCoreNum);
    return tiling;
}

static aclblasStatus_t LaunchStrmmPipeline(
    _aclblas_handle* h, const float* A, const float* B, const float* alpha, float* BOut,
    uint32_t dimA, uint32_t uM,
    const StrmmMirrorTilingData& mirrorTiling, uint32_t usedAivCoreNum,
    const StrmmGemmTilingData& gemmTiling, uint32_t usedAicCoreNum,
    const StrmmScaleTilingData& scaleTiling, uint32_t usedAivCoreNumScale,
    uint32_t uLda)
{
    size_t workspaceASize = static_cast<size_t>(uLda) * static_cast<size_t>(dimA) * STRMM_ARCH35_FP32_SIZE;
    constexpr size_t GM_ALIGN = 32;
    size_t workspaceASizeAligned = (workspaceASize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    size_t tempSize = static_cast<size_t>(uM) * static_cast<size_t>(gemmTiling.tempRowStride) * STRMM_ARCH35_FP32_SIZE;
    size_t requiredBytes = workspaceASizeAligned + tempSize;

    aclblasStatus_t wsRet = EnsureWorkspace(h, requiredBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasStrmm", "workspace ensure failed, required=%zu, ret=%d", requiredBytes, wsRet);
        return wsRet;
    }

    uint8_t* wsBase = reinterpret_cast<uint8_t*>(aclblasGetEffectiveWorkspace(h));
    uint8_t* workspaceADevice = wsBase;
    uint8_t* tempDevice = wsBase + workspaceASizeAligned;

    strmm_mirror_kernel_do(
        reinterpret_cast<const uint8_t*>(A), workspaceADevice,
        mirrorTiling, usedAivCoreNum, h->stream);
    strmm_gemm_kernel_do(
        reinterpret_cast<const uint8_t*>(workspaceADevice), reinterpret_cast<const uint8_t*>(B), tempDevice,
        gemmTiling, usedAicCoreNum, h->stream);
    strmm_scale_kernel_do(
        tempDevice, reinterpret_cast<uint8_t*>(BOut), *alpha,
        scaleTiling, usedAivCoreNumScale, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteStrmmKernels(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transA,
    aclblasDiagType_t diag,
    uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb,
    const float* A, const float* B, const float* alpha, float* BOut)
{
    uint32_t dimA = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasStrmm", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasStrmm", "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(dimA, aivCoreNum), 1);
    uint64_t tileCount = static_cast<uint64_t>(CeilDiv<uint32_t>(uM, STRMM_ARCH35_BASE_M))
                       * static_cast<uint64_t>(CeilDiv<uint32_t>(uN, STRMM_ARCH35_BASE_N));
    uint32_t usedAicCoreNum = std::max<uint32_t>(std::min<uint64_t>(tileCount, aicCoreNum), 1);
    uint32_t usedAivCoreNumScale = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);

    StrmmMirrorTilingData mirrorTiling = CalMirrorTilingData(
        usedAivCoreNum, uM, uN, uLda,
        static_cast<uint32_t>(side), static_cast<uint32_t>(uplo),
        static_cast<uint32_t>(transA), static_cast<uint32_t>(diag));
    StrmmGemmTilingData gemmTiling = CalGemmTilingData(
        usedAicCoreNum, uM, uN, static_cast<uint32_t>(side), uLda, uLdb);
    StrmmScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNumScale, uM, uN, uLdb, gemmTiling.tempRowStride);

    OP_LOGD("aclblasStrmm",
        "mirror tiling: side=%u uplo=%u trans=%u diag=%u aivCores=%u rowsPerCore=%u lda=%u dimA=%u",
        mirrorTiling.sideMode, mirrorTiling.uploMode,
        mirrorTiling.transMode, mirrorTiling.diagMode,
        mirrorTiling.usedAivCoreNum, mirrorTiling.mirrorRowsPerCore, mirrorTiling.lda, mirrorTiling.dimA);
    OP_LOGD("aclblasStrmm",
        "gemm tiling: m=%u n=%u side=%u aicCores=%u singleCoreM=%u singleCoreN=%u "
        "tileM=%u tileN=%u tileKChunk=%u lda=%u ldb=%u tempRowStride=%u",
        gemmTiling.m, gemmTiling.n, gemmTiling.sideMode, gemmTiling.usedAicCoreNum,
        gemmTiling.singleCoreM, gemmTiling.singleCoreN,
        gemmTiling.tileM, gemmTiling.tileN, gemmTiling.tileKChunk,
        gemmTiling.lda, gemmTiling.ldb, gemmTiling.tempRowStride);
    OP_LOGD("aclblasStrmm",
        "scale tiling: m=%u n=%u ldb=%u tempRowStride=%u aivCores=%u rowsPerCore=%u",
        scaleTiling.m, scaleTiling.n, scaleTiling.ldb, scaleTiling.tempRowStride,
        scaleTiling.usedAivCoreNum, scaleTiling.scaleRowsPerCore);
    OP_LOGI("aclblasStrmm", "launching mirror kernel: aivCores=%u", usedAivCoreNum);
    OP_LOGI("aclblasStrmm", "launching gemm kernel: aicCores=%u", usedAicCoreNum);
    OP_LOGI("aclblasStrmm", "launching scale kernel: aivCores=%u", usedAivCoreNumScale);

    return LaunchStrmmPipeline(h, A, B, alpha, BOut, dimA, uM,
        mirrorTiling, usedAivCoreNum, gemmTiling, usedAicCoreNum,
        scaleTiling, usedAivCoreNumScale, uLda);
}

aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t transA,
    aclblasDiagType_t diag,
    int64_t m,
    int64_t n,
    const float* alpha,
    const float* A,
    int64_t lda,
    float* B,
    int64_t ldb)
{
    CHECK_RET(m >= 0, OP_LOGE("aclblasStrmm", "m must be >= 0, got %ld", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasStrmm", "n must be >= 0, got %ld", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasStrmm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    aclblasStatus_t st = ValidateStrmmParams(side, uplo, transA, diag, m, n, lda, ldb, alpha, A, B);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    const int64_t maxU32 = static_cast<int64_t>(UINT32_MAX);
    CHECK_RET(m <= maxU32 && n <= maxU32 && lda <= maxU32 && ldb <= maxU32,
        OP_LOGE("aclblasStrmm", "dimensions exceed uint32_t limit");
        return ACLBLAS_STATUS_INVALID_VALUE);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);

    if (*alpha == 0.0f) {
        size_t bBytes = static_cast<size_t>(m) * static_cast<size_t>(ldb) * sizeof(float);
        aclError aclRet = aclrtMemsetAsync(B, bBytes, 0, bBytes, h->stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasStrmm", "aclrtMemsetAsync failed for alpha=0, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    return ExecuteStrmmKernels(h, side, uplo, transA, diag,
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb),
        A, B, alpha, B);
}
