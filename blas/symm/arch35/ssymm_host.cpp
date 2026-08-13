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
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ssymm_tiling_data.h"

struct SsymmMirrorTilingData;
void ssymm_mirror_kernel_do(uint8_t* gmA, uint8_t* gmWorkspaceA, const SsymmMirrorTilingData &tiling,
                             uint32_t numBlocks, void *stream);
struct SsymmGemmTilingData;
void ssymm_gemm_kernel_do(uint8_t* gmA, uint8_t* gmB, uint8_t* gmTemp,
                           const SsymmGemmTilingData &tiling, uint32_t numBlocks, void *stream);
struct SsymmScaleTilingData;
void ssymm_scale_kernel_do(uint8_t* gmTemp, uint8_t* gmC, const uint8_t* gmAlpha, const uint8_t* gmBeta,
                               const SsymmScaleTilingData &tiling, uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateSsymmParams(
    aclblasSideMode_t side, int m, int n,
    int lda, int ldb, int ldc, const float* alpha, const float* beta,
    const float* A, const float* B)
{
    int dimA = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    int minLda = std::max(1, dimA);
    int minLdb = std::max(1, m);
    int minLdc = std::max(1, m);
    CHECK_RET(
        lda >= minLda,
        OP_LOGE("aclblasSsymm", "lda must be >= max(1, dimA), got lda=%d, dimA=%d", lda, dimA);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= minLdb,
        OP_LOGE("aclblasSsymm", "ldb must be >= max(1, m), got ldb=%d, m=%d", ldb, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= minLdc,
        OP_LOGE("aclblasSsymm", "ldc must be >= max(1, m), got ldc=%d, m=%d", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSsymm", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSsymm", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSsymm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(B != nullptr, OP_LOGE("aclblasSsymm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
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
    uint32_t mirrorWork = tiling.mirrorRowsPerCore * dimA;
    tiling.nthreads = std::min(CeilAlign<uint32_t>(mirrorWork, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
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
    uint32_t tempRowStride, float alphaVal, float betaVal, uint32_t skipTemp,
    uint32_t alphaIsDevice, uint32_t betaIsDevice)
{
    SsymmScaleTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempRowStride = tempRowStride;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.scaleRowsPerCore = CeilDiv<uint32_t>(m, usedAivCoreNum);
    tiling.alphaVal = alphaVal;
    tiling.betaVal = betaVal;
    tiling.skipTemp = skipTemp;
    tiling.alphaIsDevice = alphaIsDevice;
    tiling.betaIsDevice = betaIsDevice;
    uint32_t scaleWork = tiling.scaleRowsPerCore * n;
    tiling.nthreads = std::min(CeilAlign<uint32_t>(scaleWork, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    return tiling;
}

// ==========================================================================
// Pointer location query: determine whether an alpha/beta scalar pointer lives
// on the host or on the device (NPU HBM). Mirrors the aclrtPointerGetAttributes
// usage in axpy_ex/scalex/srot. On query failure the call is rejected
// (consistent with the benchmarks). The two pointers are independent: alpha on
// host + beta on device (or any mix) is legal per BLAS host-or-device semantics.
// ==========================================================================
static aclblasStatus_t SsymmCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsymm", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ResolveSsymmScalars(const float* alpha, const float* beta,
    float* alphaVal, float* betaVal, bool* alphaIsDevice, bool* betaIsDevice)
{
    aclblasStatus_t alphaLocSt = SsymmCheckPtrLocation(alpha, alphaIsDevice);
    if (alphaLocSt != ACLBLAS_STATUS_SUCCESS) {
        return alphaLocSt;
    }
    aclblasStatus_t betaLocSt = SsymmCheckPtrLocation(beta, betaIsDevice);
    if (betaLocSt != ACLBLAS_STATUS_SUCCESS) {
        return betaLocSt;
    }
    *alphaVal = *alphaIsDevice ? 0.0f : (*alpha);
    *betaVal = *betaIsDevice ? 0.0f : (*beta);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// HandleSsymmAlphaZero — C = beta * C when alpha == 0 (host-only fast path).
//   Requires BOTH alpha and beta to be host pointers so their values are
//   readable on the host. When alpha (or beta) is a device pointer the host
//   cannot evaluate alpha==0 / beta==0, so this fast path is skipped by the
//   caller and the full mirror+gemm+scale pipeline runs instead — the scale
//   kernel reads alpha/beta from GM and produces the correct beta*C result.
//   beta == 0 → memset C to zero
//   beta == 1 → C unchanged, return immediately
//   otherwise → launch scale kernel with skipTemp=1 (beta*C only, temp not read)
// ==========================================================================
static aclblasStatus_t HandleSsymmAlphaZero(
    aclblasHandle_t handle, uint32_t m, uint32_t n, uint32_t ldc, float betaVal, float* C)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    if (betaVal == 0.0f) {
        aclError syncRet = aclrtSynchronizeStream(h->stream);
        if (syncRet != ACL_SUCCESS) {
            OP_LOGE("aclblasSsymm", "aclrtSynchronizeStream failed, ret=%d", syncRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * SSYMM_ARCH35_FP32_SIZE;
        aclError ret = aclrtMemset(C, cBytes, 0, cBytes);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasSsymm", "aclrtMemset failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsymm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(m, aivCoreNum), 1);
    // Host-only fast path: alpha/beta scalars carried in tiling, no GM pointer.
    SsymmScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNum, m, n, ldc, ldc, 0.0f, betaVal, 1U, 0U, 0U);
    OP_LOGI("aclblasSsymm", "alpha==0 fast path: launching beta-scale kernel, aivCores=%u", usedAivCoreNum);
    ssymm_scale_kernel_do(
        reinterpret_cast<uint8_t*>(C), reinterpret_cast<uint8_t*>(C),
        nullptr, nullptr,
        scaleTiling, usedAivCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchSsymmPipeline(
    _aclblas_handle* h, const float* A, const float* B, float* C,
    uint32_t dimA, uint32_t uM, uint32_t uN,
    const SsymmMirrorTilingData& mirrorTiling, uint32_t usedAivCoreNum,
    const SsymmGemmTilingData& gemmTiling, uint32_t usedAicCoreNum,
    const SsymmScaleTilingData& scaleTiling, uint32_t usedAivCoreNumScale,
    uint32_t uLda, const float* alphaPtr, const float* betaPtr)
{
    size_t workspaceASize = static_cast<size_t>(uLda) * static_cast<size_t>(dimA) * SSYMM_ARCH35_FP32_SIZE;
    size_t tempSize = static_cast<size_t>(uN) * static_cast<size_t>(gemmTiling.tempRowStride) * SSYMM_ARCH35_FP32_SIZE;
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
    // Forward device alpha/beta pointers verbatim; host-mode forwards nullptr
    // (kernel reads the tiling scalar instead). See SsymmScaleTilingData flags.
    ssymm_scale_kernel_do(
        tempDevice, (uint8_t*)C,
        reinterpret_cast<const uint8_t*>(alphaPtr),
        reinterpret_cast<const uint8_t*>(betaPtr),
        scaleTiling, usedAivCoreNumScale, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteSsymmKernels(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
    uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const float* A, const float* B, float alphaVal, float betaVal, float* C,
    const float* alphaPtr, const float* betaPtr,
    uint32_t alphaIsDevice, uint32_t betaIsDevice)
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
    // Column-major adaptation: GEMM kernel internally works in row-major. To compute the
    // column-major result C(m×n), the GEMM kernel computes C^T (row-major n×m) by swapping
    // m↔n and flipping side. This leverages two properties:
    //   1. A_sym is symmetric: row-major and column-major storage are identical for a full
    //      symmetric matrix, so the mirror-filled workspaceA is read correctly either way.
    //   2. Column-major B(m×n) in memory == row-major B^T(n×m): the row-major GEMM kernel
    //      reads B as B^T, which is exactly what C^T = ... * B^T needs.
    // After the swap, side=LEFT (A_sym on the left) becomes side=RIGHT (A_sym on the right)
    // in the kernel's view, and vice versa. lda/ldb are NOT swapped: the GEMM kernel always
    // accesses aGlobal (workspaceA) with tiling.lda and bGlobal (B) with tiling.ldb,
    // regardless of which side each matrix is on (see SsymmCopyInA1/SsymmCopyInB1 in
    // ssymm_kernel.cpp). Since aGlobal always holds A and bGlobal always holds B, passing
    // the original uLda/uLdb directly gives each matrix its correct leading dimension.
    uint32_t gemmSide = (side == ACLBLAS_SIDE_LEFT) ? ACLBLAS_SIDE_RIGHT : ACLBLAS_SIDE_LEFT;
    SsymmGemmTilingData gemmTiling = CalGemmTilingData(
        usedAicCoreNum, uN, uM, gemmSide, uLda, uLdb, uLdc);
    SsymmScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNumScale, uM, uN, uLdc, gemmTiling.tempRowStride, alphaVal, betaVal, 0U,
        alphaIsDevice, betaIsDevice);

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
        "scale tiling: m=%u n=%u ldc=%u tempRowStride=%u aivCores=%u rowsPerCore=%u "
        "alphaIsDevice=%u betaIsDevice=%u",
        scaleTiling.m, scaleTiling.n, scaleTiling.ldc, scaleTiling.tempRowStride,
        scaleTiling.usedAivCoreNum, scaleTiling.scaleRowsPerCore,
        scaleTiling.alphaIsDevice, scaleTiling.betaIsDevice);
    OP_LOGI("aclblasSsymm", "launching mirror kernel: aivCores=%u", usedAivCoreNum);
    OP_LOGI("aclblasSsymm", "launching gemm kernel: aicCores=%u", usedAicCoreNum);
    OP_LOGI("aclblasSsymm", "launching scale kernel: aivCores=%u alphaIsDevice=%u betaIsDevice=%u",
        usedAivCoreNumScale, alphaIsDevice, betaIsDevice);

    return LaunchSsymmPipeline(h, A, B, C, dimA, uM, uN,
        mirrorTiling, usedAivCoreNum, gemmTiling, usedAicCoreNum,
        scaleTiling, usedAivCoreNumScale, uLda, alphaPtr, betaPtr);
}

aclblasStatus_t aclblasSsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc)
{
    // Problem 2: correct validation order — handle, enums, m/n before quick return.
    // This ensures invalid handle/enums are caught even when m==0 or n==0.
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasSsymm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE("aclblasSsymm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsymm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSsymm", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSsymm", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);

    // Quick return: m==0 or n==0 — no pointer/ld validation needed (BLAS standard).
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Validate ld and non-C pointers (side/uplo already checked above).
    // C is validated later based on beta value (BLAS: C may be NULL when beta==0).
    // int is 32-bit signed; non-negative int never exceeds UINT32_MAX, so no overflow check needed.
    aclblasStatus_t st = ValidateSsymmParams(side, m, n, lda, ldb, ldc,
        alpha, beta, A, B);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Resolve alpha/beta scalars before checking C: if beta is a host pointer
    // and beta==0, C is not required to be a valid pointer (BLAS standard).
    float alphaVal, betaVal;
    bool alphaIsDevice, betaIsDevice;
    aclblasStatus_t scalarSt = ResolveSsymmScalars(alpha, beta,
        &alphaVal, &betaVal, &alphaIsDevice, &betaIsDevice);
    if (scalarSt != ACLBLAS_STATUS_SUCCESS) {
        return scalarSt;
    }

    // BLAS: if beta == 0 then C does not have to be a valid input.
    // Device beta cannot be evaluated on host — require C != nullptr and let
    // the scale kernel handle beta==0 (skip reading C).
    if (!(!betaIsDevice && betaVal == 0.0f)) {
        CHECK_RET(C != nullptr, OP_LOGE("aclblasSsymm", "C must not be nullptr when beta != 0");
            return ACLBLAS_STATUS_INVALID_VALUE);
    }

    // alpha==0 fast path: C = beta * C, skip mirror+gemm. This optimization needs BOTH
    // alpha and beta readable on the host (the fast path branches on betaVal==0/1 and
    // may memset/scale C directly on the host side). When alpha is a device pointer the
    // host cannot know alpha==0; when beta is a device pointer the host cannot branch on
    // beta. In either case, skip the fast path and run the full mirror+gemm+scale pipeline
    // — the scale kernel resolves alpha/beta from GM and produces the correct result
    // (including the alpha==0 → beta*C case, just without the skip-mirror+gemm optimization).
    if (!alphaIsDevice && !betaIsDevice && alphaVal == 0.0f) {
        return HandleSsymmAlphaZero(handle, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
            static_cast<uint32_t>(ldc), betaVal, C);
    }

    // Forward device pointers verbatim to the kernel; host mode forwards nullptr so the
    // kernel reads the tiling scalar instead (alphaIsDevice/betaIsDevice gate the read).
    const float* alphaFwd = alphaIsDevice ? alpha : nullptr;
    const float* betaFwd = betaIsDevice ? beta : nullptr;

    return ExecuteSsymmKernels(handle, side, uplo,
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc),
        A, B, alphaVal, betaVal, C,
        alphaFwd, betaFwd,
        alphaIsDevice ? 1U : 0U, betaIsDevice ? 1U : 0U);
}
