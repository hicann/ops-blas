/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyrk_host.cpp
 * \brief SSYRK Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "ssyrk_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static inline GM_ADDR ToGmAddr(const float* p)
{
    return reinterpret_cast<GM_ADDR>(const_cast<float*>(p));
}

static aclblasStatus_t ValidateSsyrkParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* C, int ldc)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsyrk", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSsyrk", "trans must be OP_N, OP_T or OP_C, got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    if (trans == ACLBLAS_OP_N) {
        CHECK_RET(
            lda >= std::max(1, n),
            OP_LOGE("aclblasSsyrk", "lda must be >= max(1,n) for trans=N, got lda=%d n=%d", lda, n);
            return ACLBLAS_STATUS_INVALID_VALUE);
    } else {
        CHECK_RET(
            lda >= std::max(1, k),
            OP_LOGE("aclblasSsyrk", "lda must be >= max(1,k) for trans=T, got lda=%d k=%d", lda, k);
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(
        ldc >= std::max(1, n),
        OP_LOGE("aclblasSsyrk", "ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr,
        OP_LOGE("aclblasSsyrk", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr,
        OP_LOGE("aclblasSsyrk", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        !(A == nullptr && n > 0 && k > 0),
        OP_LOGE("aclblasSsyrk", "A must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        !(C == nullptr && n > 0),
        OP_LOGE("aclblasSsyrk", "C must not be nullptr when n>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SyrkGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t n, uint32_t k, uint32_t lda, uint8_t isTransN,
    uint32_t tempRowStride)
{
    SyrkGemmTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.isTransN = isTransN;

    uint32_t coreDim = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.singleCoreM = std::max<uint32_t>(coreDim, SYRK_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreDim, SYRK_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(SYRK_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(SYRK_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = SYRK_ARCH35_DEFAULT_TILE_K_CHUNK;

    if (n < SYRK_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(n, SYRK_ARCH35_BASE_M));
    }
    if (n < SYRK_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(n, SYRK_ARCH35_BASE_N));
    }

    uint32_t alignedM = CeilAlign<uint32_t>(tileM, SYRK_ARCH35_BASE_M);
    uint32_t alignedN = CeilAlign<uint32_t>(tileN, SYRK_ARCH35_BASE_N);
    uint32_t l1Budget = SYRK_ARCH35_L1_SIZE_BYTES * SYRK_ARCH35_L1_USAGE_RATIO_NUM
                       / SYRK_ARCH35_L1_USAGE_RATIO_DEN;
    uint32_t denom = SYRK_ARCH35_L1_BUF_NUM * SYRK_ARCH35_FP32_SIZE * (alignedM + alignedN);
    uint32_t maxAlignedK = (denom > 0) ? (l1Budget / denom) : 0;
    uint32_t maxKChunk = (maxAlignedK / SYRK_ARCH35_BASE_K) * SYRK_ARCH35_BASE_K;
    maxKChunk = std::max<uint32_t>(maxKChunk, SYRK_ARCH35_BASE_K);
    if (tileKChunk > maxKChunk) {
        tileKChunk = maxKChunk;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.tempRowStride = tempRowStride;

    return tiling;
}

static SyrkScaleTilingData CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc,
    uint8_t isAlphaZero, uint8_t isKZero, uint8_t isBetaZero, uint32_t tempRowStride,
    uint8_t uploMode, float alphaVal, float betaVal)
{
    SyrkScaleTilingData tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempRowStride = tempRowStride;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.alphaVal = alphaVal;
    tiling.betaVal = betaVal;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    return tiling;
}

static aclblasStatus_t ReadAlphaBetaFromDevice(
    const float* alpha, const float* beta, float& alphaVal, float& betaVal,
    aclrtStream stream)
{
    alphaVal = 0.0f;
    betaVal = 0.0f;
    aclError aclRet = aclrtMemcpyAsync(&alphaVal, sizeof(float), alpha, sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyrk", "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpyAsync(&betaVal, sizeof(float), beta, sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyrk", "aclrtMemcpyAsync beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyrk", "aclrtSynchronizeStream for D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static uint32_t GetUsedAivCoreNum(uint32_t n, uint32_t aivCoreNum)
{
    return std::max<uint32_t>(std::min<uint32_t>(n, aivCoreNum), 1);
}

static uint32_t GetUsedAicCoreNum(uint32_t n)
{
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasSsyrk", "cube core count is 0");
        return 0;
    }
    uint32_t coreDim = CeilDiv<uint32_t>(n, aicCoreNum);
    uint32_t singleCoreDim = std::max<uint32_t>(coreDim, SYRK_ARCH35_BASE_M);
    uint64_t tileCount = static_cast<uint64_t>(CeilDiv<uint32_t>(n, singleCoreDim))
                       * static_cast<uint64_t>(CeilDiv<uint32_t>(n, singleCoreDim));
    return std::max<uint32_t>(std::min<uint64_t>(tileCount, aicCoreNum), 1);
}

static aclblasStatus_t LaunchGemmKernel(
    _aclblas_handle* h, uint32_t n, uint32_t k, uint32_t lda,
    aclblasOperation_t trans, const float* A,
    uint32_t usedAicCoreNum, uint8_t* tempDevice, uint32_t tempRowStride)
{
    uint8_t isTransN = (trans == ACLBLAS_OP_N) ? 1 : 0;
    SyrkGemmTilingData gemmTiling = CalGemmTilingData(
        usedAicCoreNum, n, k, lda, isTransN, tempRowStride);
    OP_LOGD("aclblasSsyrk",
        "gemm tiling: n=%u k=%u lda=%u isTransN=%u aicCores=%u singleCoreM=%u singleCoreN=%u "
        "tileM=%u tileN=%u tileKChunk=%u tempRowStride=%u",
        gemmTiling.n, gemmTiling.k, gemmTiling.lda, gemmTiling.isTransN, usedAicCoreNum,
        gemmTiling.singleCoreM, gemmTiling.singleCoreN,
        gemmTiling.tileM, gemmTiling.tileN, gemmTiling.tileKChunk,
        gemmTiling.tempRowStride);
    OP_LOGI("aclblasSsyrk", "launching gemm kernel: aicCores=%u", usedAicCoreNum);

    syrk_gemm_kernel_do(
        ToGmAddr(A), tempDevice,
        gemmTiling, usedAicCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchScaleKernel(
    _aclblas_handle* h, uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc,
    bool isAlphaZero, bool isKZero, bool isBetaZero,
    aclblasFillMode_t uplo, float alphaVal, float betaVal,
    uint32_t tempRowStride, uint8_t* tempDevice, float* C)
{
    SyrkScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNum, n, ldc,
        static_cast<uint8_t>(isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(isKZero ? 1 : 0),
        static_cast<uint8_t>(isBetaZero ? 1 : 0),
        tempRowStride,
        static_cast<uint8_t>(uplo), alphaVal, betaVal);

    OP_LOGD("aclblasSsyrk",
        "scale tiling: n=%u ldc=%u aivCores=%u isAlphaZero=%u isKZero=%u isBetaZero=%u uplo=%u",
        scaleTiling.n, scaleTiling.ldc, usedAivCoreNum,
        static_cast<uint32_t>(scaleTiling.isAlphaZero),
        static_cast<uint32_t>(scaleTiling.isKZero),
        static_cast<uint32_t>(scaleTiling.isBetaZero),
        static_cast<uint32_t>(scaleTiling.uploMode));
    OP_LOGI("aclblasSsyrk", "launching scale kernel: aivCores=%u", usedAivCoreNum);

    syrk_scale_kernel_do(
        tempDevice, reinterpret_cast<uint8_t*>(C),
        scaleTiling, usedAivCoreNum, h->stream);
}

static aclblasStatus_t LaunchSsyrkKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans,
    uint32_t n, uint32_t k,
    const float* alpha, const float* A, uint32_t lda,
    const float* beta, float* C, uint32_t ldc)
{
    float alphaVal = 0.0f;
    float betaVal = 0.0f;
    aclblasStatus_t readRet = ReadAlphaBetaFromDevice(alpha, beta, alphaVal, betaVal, h->stream);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    bool isAlphaZero = (alphaVal == 0.0f);
    bool isKZero = (k == 0);
    bool isBetaZero = (betaVal == 0.0f);

    OP_LOGD("aclblasSsyrk",
        "alpha=%.6f beta=%.6f isAlphaZero=%d isKZero=%d isBetaZero=%d",
        alphaVal, betaVal, isAlphaZero, isKZero, isBetaZero);

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsyrk", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t usedAivCoreNum = GetUsedAivCoreNum(n, aivCoreNum);
    uint32_t usedAicCoreNum = GetUsedAicCoreNum(n);
    if (usedAicCoreNum == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t tempRowStride = CeilAlign<uint32_t>(n, SYRK_ARCH35_FIXPIPE_N_ALIGN);
    constexpr size_t GM_ALIGN = 32;
    size_t tempSize = static_cast<size_t>(n) * static_cast<size_t>(tempRowStride) * sizeof(float);
    size_t requiredBytes = (tempSize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, requiredBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSsyrk", "workspace ensure failed, required=%zu, ret=%d", requiredBytes, wsRet);
        return wsRet;
    }

    uint8_t* tempDevice = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    if (!isAlphaZero && !isKZero) {
        aclblasStatus_t gemmRet = LaunchGemmKernel(h, n, k, lda, trans, A,
            usedAicCoreNum, tempDevice, tempRowStride);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    LaunchScaleKernel(h, usedAivCoreNum, n, ldc,
        isAlphaZero, isKZero, isBetaZero, uplo, alphaVal, betaVal,
        tempRowStride, tempDevice, C);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const float* A, int lda, const float* beta, float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSsyrk", "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(n >= 0,
        OP_LOGE("aclblasSsyrk", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0,
        OP_LOGE("aclblasSsyrk", "k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateSsyrkParams(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    return LaunchSsyrkKernel(h, uplo, trans,
        static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        alpha, A, static_cast<uint32_t>(lda),
        beta, C, static_cast<uint32_t>(ldc));
}
