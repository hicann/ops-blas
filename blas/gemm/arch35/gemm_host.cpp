/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include <algorithm>
#include <cstdint>
#include <complex>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ==========================================================================
//  Tiling computation (mirrors gemm_ex for FP32)
// ==========================================================================

static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum, int m, int n)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (n + tiling.baseN - 1) / tiling.baseN;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1)
            nb = 1;
        int32_t utilization = mb * nb;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    if (bestUtilization == 0) {
        bestMBlocks = 1;
        bestNBlocks = 1;
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static GemmTilingData CalGemmTilingData(
    int m, int n, int k, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    float alphaReal, float alphaImag, float betaReal, float betaImag)
{
    GemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.cLdc = ldc;
    tiling.baseM = GEMM_BASE_M;
    tiling.baseN = GEMM_BASE_N;
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaReal = betaReal;
    tiling.betaImag = betaImag;
    tiling.hasBeta = (betaReal != 0.0f || betaImag != 0.0f) ? 1 : 0;
    return tiling;
}

static void ApplyColMajorSwap(GemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyColMajorSwap(tiling);
    CalcMultiCorePartition(tiling, cubeCoreNum, tiling.m, tiling.n);
}

// ==========================================================================
//  ValidateGemmParams — common parameter validation
// ==========================================================================
static aclblasStatus_t ValidateGemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, int lda, int ldb, const void* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasGemm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE("aclblasGemm", "invalid transa: %d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE("aclblasGemm", "invalid transb: %d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasGemm", "m, n, k must be non-negative: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int ldaMin = isTransA ? std::max(1, k) : std::max(1, m);
    if (lda < ldaMin) {
        OP_LOGE("aclblasGemm", "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = isTransB ? std::max(1, n) : std::max(1, k);
    if (ldb < ldbMin) {
        OP_LOGE("aclblasGemm", "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasGemm", "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasGemm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasGemm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGemmPointers(
    int k, float alphaAbs, const void* A, const void* B, float betaAbs, const void* C)
{
    if (k > 0 && alphaAbs != 0.0f) {
        if (A == nullptr) {
            OP_LOGE("aclblasGemm", "A must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            OP_LOGE("aclblasGemm", "B must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (k > 0) {
        if (C == nullptr) {
            OP_LOGE("aclblasGemm", "C must not be nullptr when k > 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (betaAbs != 0.0f && C == nullptr) {
        OP_LOGE("aclblasGemm", "C must not be nullptr when beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  HandleSgemmAlphaZero — C = beta * C when k == 0 or alpha == 0
// ==========================================================================
static aclblasStatus_t HandleSgemmAlphaZero(aclblasHandle_t handle, int m, int n, int ldc, float betaVal, float* C)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(float);
        aclError ret = aclrtMemset(C, cBytes, 0, cBytes);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasSgemm", "aclrtMemset failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    gemm_scale_do(vecBlocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, betaVal, 0.0f, 0);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  LaunchSgemmKernel — FP32 device kernel launch
// ==========================================================================
static aclblasStatus_t PrepareSgemmWorkspace(
    _aclblas_handle* h, bool needPostProcess, int m, int n, uint8_t*& tempABDevice)
{
    tempABDevice = nullptr;
    if (!needPostProcess) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t workspaceNeed = static_cast<size_t>(CeilAlign(m, GEMM_FRACTAL)) * n * sizeof(float);
    if (!CheckEffectiveWorkspaceSize(h, workspaceNeed)) {
        OP_LOGE("aclblasSgemm", "workspace need %zu bytes", workspaceNeed);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    tempABDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (reinterpret_cast<uintptr_t>(tempABDevice) % 512 != 0) {
        OP_LOGW("aclblasSgemm", "workspace address not 512B aligned: %p", tempABDevice);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchAlphaBetaKernel(
    aclrtStream stream, uint8_t* tempABDevice, float* C, int n, const GemmTilingData& abTiling)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    GemmTilingData tiling = abTiling;
    tiling.usedCoreNum = static_cast<int32_t>(vecBlocks);
    OP_LOGI("aclblasSgemm", "launching alpha_beta kernel: aivBlocks=%u", vecBlocks);
    gemm_alpha_beta_do(
        vecBlocks, stream, tempABDevice, reinterpret_cast<uint8_t*>(C), reinterpret_cast<uint8_t*>(C), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchSgemmKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, float alphaVal,
    const float* A, int lda, const float* B, int ldb, float betaVal, float* C, int ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (k == 0 || alphaVal == 0.0f) {
        return HandleSgemmAlphaZero(handle, m, n, ldc, betaVal, C);
    }

    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);

    GemmTilingData tiling = CalGemmTilingData(m, n, k, lda, ldb, ldc, transa, transb, alphaVal, 0.0f, betaVal, 0.0f);

    if (needPostProcess) {
        tiling.ldc = static_cast<int32_t>(CeilAlign(m, GEMM_FRACTAL));
    }
    GemmTilingData abTiling = tiling;

    PrepareCubeTiling(tiling, cubeCoreNum);

    uint32_t numBlocks = static_cast<uint32_t>(tiling.usedCoreNum);
    uint8_t* aDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* bDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));

    uint8_t* tempABDevice = nullptr;
    aclblasStatus_t st = PrepareSgemmWorkspace(h, needPostProcess, m, n, tempABDevice);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    uint8_t* cDevicePtr = needPostProcess ? tempABDevice : reinterpret_cast<uint8_t*>(C);

    OP_LOGI("aclblasSgemm", "launching cube kernel: blocks=%u", numBlocks);
    gemm_kernel_do(numBlocks, stream, aDevicePtr, bDevicePtr, cDevicePtr, tiling);

    if (needPostProcess) {
        return LaunchAlphaBetaKernel(stream, tempABDevice, C, n, abTiling);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Complex32 helpers
// ==========================================================================
struct ComplexBuffers {
    std::vector<std::complex<float>> hA;
    std::vector<std::complex<float>> hB;
    std::vector<std::complex<float>> hC;
    std::vector<float> reA;
    std::vector<float> imA;
    std::vector<float> reB;
    std::vector<float> imB;
    std::vector<float> t1;
    std::vector<float> t2;
    std::vector<float> t3;
    std::vector<float> t4;
    void *d_reA = nullptr, *d_imA = nullptr, *d_reB = nullptr, *d_imB = nullptr;
    void *d_t1 = nullptr, *d_t2 = nullptr, *d_t3 = nullptr, *d_t4 = nullptr;
};

static void FreeComplexBuffers(ComplexBuffers& buf)
{
    auto safeFree = [](void* p, const char* name) {
        if (p != nullptr) {
            aclError ret = aclrtFree(p);
            if (ret != ACL_SUCCESS) {
                OP_LOGW("aclblasCgemm", "aclrtFree(%s) failed, ret=%d", name, ret);
            }
        }
    };
    safeFree(buf.d_reA, "d_reA");
    safeFree(buf.d_imA, "d_imA");
    safeFree(buf.d_reB, "d_reB");
    safeFree(buf.d_imB, "d_imB");
    safeFree(buf.d_t1, "d_t1");
    safeFree(buf.d_t2, "d_t2");
    safeFree(buf.d_t3, "d_t3");
    safeFree(buf.d_t4, "d_t4");
    buf.d_reA = nullptr;
    buf.d_imA = nullptr;
    buf.d_reB = nullptr;
    buf.d_imB = nullptr;
    buf.d_t1 = nullptr;
    buf.d_t2 = nullptr;
    buf.d_t3 = nullptr;
    buf.d_t4 = nullptr;
}

static void DeinterleaveMatrix(
    const std::complex<float>* hMat, int physRows, int physCols, int ld, int logicalRows, bool transFlag, bool conjFlag,
    float* reOut, float* imOut)
{
    for (int j = 0; j < physCols; j++) {
        for (int i = 0; i < physRows; i++) {
            std::complex<float> val = hMat[i + j * ld];
            if (conjFlag)
                val = std::conj(val);
            int lr = transFlag ? j : i;
            int lc = transFlag ? i : j;
            reOut[lr + lc * logicalRows] = val.real();
            imOut[lr + lc * logicalRows] = val.imag();
        }
    }
}

static aclblasStatus_t HandleCgemmAlphaZero(
    aclblasHandle_t handle, int m, int n, int ldc, std::complex<float> betaVal, aclblasComplex* C)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(std::complex<float>);
        aclError ret = aclrtMemset(C, cBytes, 0, cBytes);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCgemm", "aclrtMemset failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == std::complex<float>(1.0f, 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    gemm_scale_do(vecBlocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, betaVal.real(), betaVal.imag(), 1);
    return ACLBLAS_STATUS_SUCCESS;
}

static void CombineComplexResults(
    float ar, float ai, float br, float bi, const float* t1, const float* t2, const float* t3, const float* t4,
    std::complex<float>* hC, int m, int n, int ldc, int tempLdc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int idx = i + j * tempLdc;
            float ab_real = t1[idx] - t2[idx];
            float ab_imag = t3[idx] + t4[idx];
            float c_real = hC[i + j * ldc].real();
            float c_imag = hC[i + j * ldc].imag();
            hC[i + j * ldc] = std::complex<float>(
                ar * ab_real - ai * ab_imag + br * c_real - bi * c_imag,
                ar * ab_imag + ai * ab_real + br * c_imag + bi * c_real);
        }
    }
}

// ==========================================================================
//  LaunchCgemmKernel — Complex32 device kernel launch
// ==========================================================================

struct MallocItem {
    void** ptr;
    size_t bytes;
    const char* name;
};
struct MemcpyItem {
    void* dst;
    const void* src;
    size_t bytes;
    aclrtMemcpyKind kind;
    const char* name;
};

static aclblasStatus_t CheckedMallocBatch(MallocItem* items, size_t count, ComplexBuffers& buf)
{
    for (size_t i = 0; i < count; i++) {
        aclError ret = aclrtMalloc(items[i].ptr, items[i].bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCgemm", "aclrtMalloc %s failed, ret=%d", items[i].name, ret);
            FreeComplexBuffers(buf);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CheckedMemcpyBatch(MemcpyItem* items, size_t count, ComplexBuffers& buf)
{
    for (size_t i = 0; i < count; i++) {
        aclError ret = aclrtMemcpy(items[i].dst, items[i].bytes, items[i].src, items[i].bytes, items[i].kind);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCgemm", "aclrtMemcpy %s failed, ret=%d", items[i].name, ret);
            FreeComplexBuffers(buf);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCgemmKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    std::complex<float> alphaVal, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    std::complex<float> betaVal, aclblasComplex* C, int ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasCgemm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (k == 0 || alphaVal == std::complex<float>(0.0f, 0.0f)) {
        return HandleCgemmAlphaZero(handle, m, n, ldc, betaVal, C);
    }

    int physRowsA = (transa == ACLBLAS_OP_N) ? m : k;
    int physColsA = (transa == ACLBLAS_OP_N) ? k : m;
    int physRowsB = (transb == ACLBLAS_OP_N) ? k : n;
    int physColsB = (transb == ACLBLAS_OP_N) ? n : k;

    size_t aCount = static_cast<size_t>(lda) * physColsA;
    size_t bCount = static_cast<size_t>(ldb) * physColsB;
    size_t mkCount = static_cast<size_t>(m) * k;
    size_t knCount = static_cast<size_t>(k) * n;
    size_t tempLdc = static_cast<size_t>(CeilAlign(m, GEMM_FRACTAL));
    size_t mnCount = tempLdc * n;
    size_t aBytes = aCount * sizeof(std::complex<float>);
    size_t bBytes = bCount * sizeof(std::complex<float>);
    size_t reABytes = mkCount * sizeof(float);
    size_t reBBytes = knCount * sizeof(float);
    size_t mnBytes = mnCount * sizeof(float);

    ComplexBuffers buf;
    buf.hA.resize(aCount);
    buf.hB.resize(bCount);
    buf.reA.resize(mkCount);
    buf.imA.resize(mkCount);
    buf.reB.resize(knCount);
    buf.imB.resize(knCount);

    MemcpyItem d2hInit[] = {
        {buf.hA.data(), A, aBytes, ACL_MEMCPY_DEVICE_TO_HOST, "hA"},
        {buf.hB.data(), B, bBytes, ACL_MEMCPY_DEVICE_TO_HOST, "hB"},
    };
    aclblasStatus_t st = CheckedMemcpyBatch(d2hInit, 2, buf);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    DeinterleaveMatrix(
        buf.hA.data(), physRowsA, physColsA, lda, m, transa != ACLBLAS_OP_N, transa == ACLBLAS_OP_C, buf.reA.data(),
        buf.imA.data());
    DeinterleaveMatrix(
        buf.hB.data(), physRowsB, physColsB, ldb, k, transb != ACLBLAS_OP_N, transb == ACLBLAS_OP_C, buf.reB.data(),
        buf.imB.data());

    constexpr size_t matrixABufCount = 2; // reA + imA
    constexpr size_t matrixBBufCount = 2; // reB + imB
    constexpr size_t tempResultCount = 4; // t1, t2, t3, t4
    size_t totalWorkspace = reABytes * matrixABufCount + reBBytes * matrixBBufCount + mnBytes * tempResultCount;
    if (!CheckEffectiveWorkspaceSize(h, totalWorkspace)) {
        OP_LOGE("aclblasCgemm", "workspace need %zu bytes", totalWorkspace);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint8_t* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* d_reA = workspace;
    uint8_t* d_imA = d_reA + reABytes;
    uint8_t* d_reB = d_imA + reABytes;
    uint8_t* d_imB = d_reB + reBBytes;
    uint8_t* d_t1 = d_imB + reBBytes;
    uint8_t* d_t2 = d_t1 + mnBytes;
    uint8_t* d_t3 = d_t2 + mnBytes;
    uint8_t* d_t4 = d_t3 + mnBytes;

    MemcpyItem h2dItems[] = {
        {d_reA, buf.reA.data(), reABytes, ACL_MEMCPY_HOST_TO_DEVICE, "d_reA"},
        {d_imA, buf.imA.data(), reABytes, ACL_MEMCPY_HOST_TO_DEVICE, "d_imA"},
        {d_reB, buf.reB.data(), reBBytes, ACL_MEMCPY_HOST_TO_DEVICE, "d_reB"},
        {d_imB, buf.imB.data(), reBBytes, ACL_MEMCPY_HOST_TO_DEVICE, "d_imB"},
    };
    st = CheckedMemcpyBatch(h2dItems, 4, buf);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    GemmTilingData cubeTiling = CalGemmTilingData(
        m, n, k, m, k, m, ACLBLAS_OP_N, ACLBLAS_OP_N, alphaVal.real(), alphaVal.imag(), betaVal.real(), betaVal.imag());
    PrepareCubeTiling(cubeTiling, cubeCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(CeilAlign(m, GEMM_FRACTAL));

    uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);
    OP_LOGI("aclblasCgemm", "launching 4 cube + combine: aicBlocks=%u", numBlocks);

    gemm_kernel_do(numBlocks, stream, d_reB, d_reA, d_t1, cubeTiling);
    gemm_kernel_do(numBlocks, stream, d_imB, d_imA, d_t2, cubeTiling);
    gemm_kernel_do(numBlocks, stream, d_imB, d_reA, d_t3, cubeTiling);
    gemm_kernel_do(numBlocks, stream, d_reB, d_imA, d_t4, cubeTiling);

    uint32_t combineBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    gemm_cgemm_combine_do(
        combineBlocks, stream, d_t1, d_t2, d_t3, d_t4, static_cast<int32_t>(tempLdc), reinterpret_cast<uint8_t*>(C), m,
        n, ldc, alphaVal.real(), alphaVal.imag(), betaVal.real(), betaVal.imag());

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  aclblasSgemm — public API entry (FP32)
// ==========================================================================
extern "C" aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    OP_LOGI(
        "aclblasSgemm", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k);

    aclblasStatus_t st = ValidateGemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    float betaVal = *beta;

    float alphaAbs = std::abs(alphaVal);
    st = ValidateGemmPointers(k, alphaAbs, A, B, betaVal, C);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    return LaunchSgemmKernel(handle, transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
}

// ==========================================================================
//  aclblasCgemm — public API entry (Complex32)
// ==========================================================================
extern "C" aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    OP_LOGI(
        "aclblasCgemm", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k);

    aclblasStatus_t st = ValidateGemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    std::complex<float> alphaVal(alpha->real, alpha->imag);
    std::complex<float> betaVal(beta->real, beta->imag);

    float betaAbs = std::abs(betaVal.real()) + std::abs(betaVal.imag());
    float alphaAbs = std::abs(alphaVal.real()) + std::abs(alphaVal.imag());
    st = ValidateGemmPointers(k, alphaAbs, A, B, betaAbs, C);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    return LaunchCgemmKernel(handle, transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
}

#endif
