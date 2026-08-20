/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <algorithm>
#include <limits>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "chemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateChemmParams(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    bool* quickReturn)
{
    if (quickReturn != nullptr) {
        *quickReturn = false;
    }
    if (handle == nullptr) {
        OP_LOGE("aclblasChemm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasChemm", "side must be LEFT or RIGHT");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasChemm", "uplo must be UPPER or LOWER");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0) {
        OP_LOGE("aclblasChemm", "m and n must be >= 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        if (quickReturn != nullptr) {
            *quickReturn = true;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmDataParams(
    aclblasSideMode_t side, int64_t m, int64_t n, const aclblasComplex* alpha, const aclblasComplex* A,
    int64_t lda, const aclblasComplex* B, int64_t ldb, const aclblasComplex* beta, aclblasComplex* C,
    int64_t ldc)
{
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr) {
        OP_LOGE("aclblasChemm", "null pointer in data params");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max<int64_t>(1, aDim) || ldb < std::max<int64_t>(1, n) ||
        ldc < std::max<int64_t>(1, n)) {
        OP_LOGE("aclblasChemm", "leading dimension too small");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int64_t u32Max = static_cast<int64_t>(UINT32_MAX);
    if (m > u32Max || n > u32Max || aDim > u32Max || lda > u32Max || ldb > u32Max || ldc > u32Max) {
        OP_LOGE("aclblasChemm", "dimensions exceed uint32_t limit");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static bool IsChemmWorkspaceSizeSafe(uint32_t m, uint32_t n, uint32_t k)
{
    constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t maxBytes = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    uint64_t mk = static_cast<uint64_t>(m) * k;
    uint64_t kn = static_cast<uint64_t>(k) * n;
    uint64_t mn = static_cast<uint64_t>(m) * n;
    if (mk > maxValue / 2U || kn > maxValue / 2U || mn > maxValue / 4U) {
        return false;
    }
    uint64_t totalFloats = 2U * mk + 2U * kn + 4U * mn;
    return totalFloats <= (maxBytes - 31U) / sizeof(float);
}

static aclblasStatus_t ConfigureChemmTiling(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, int64_t lda, int64_t ldb, const aclblasComplex* beta, int64_t ldc,
    ChemmMmadTiling& tiling)
{
    uint32_t aivCores = GetAivCoreCount();
    uint32_t aicCores = GetAicCoreCount();
    if (aivCores == 0 || aicCores == 0) {
        OP_LOGE("aclblasChemm", "core count query failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t uM = static_cast<uint32_t>(m);
    uint32_t uN = static_cast<uint32_t>(n);
    uint32_t uK = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;
    tiling = {};
    tiling.m = uM;
    tiling.n = uN;
    tiling.k = uK;
    tiling.sideMode = static_cast<uint32_t>(side);
    tiling.uploMode = static_cast<uint32_t>(uplo);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.alphaReal = alpha->real;
    tiling.alphaImag = alpha->imag;
    tiling.betaReal = beta->real;
    tiling.betaImag = beta->imag;
    uint64_t m64 = static_cast<uint64_t>(m);
    uint64_t n64 = static_cast<uint64_t>(n);
    uint64_t k64 = static_cast<uint64_t>(uK);
    bool cacheLineSafe = (m64 * k64) % 16U == 0U && (k64 * n64) % 16U == 0U &&
        (m64 * n64) % 16U == 0U && static_cast<uint64_t>(ldc) % 8U == 0U;
    tiling.aivCoreNum = cacheLineSafe ? std::max<uint32_t>(std::min<uint32_t>(uM, aivCores), 1U) : 1U;
    uint64_t cubeTileCount = static_cast<uint64_t>(uM / 64U) * (uN / 64U);
    tiling.aicCoreNum = std::min<uint64_t>(aicCores, cubeTileCount);
    if (!IsChemmWorkspaceSizeSafe(uM, uN, uK)) {
        OP_LOGE("aclblasChemm", "workspace size calculation overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t requiredBytes = CalcChemmWorkspaceBytes(uM, uN, uK);
    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    if (requiredBytes > availableBytes) {
        OP_LOGE("aclblasChemm", "workspace too small: required=%zu, available=%zu", requiredBytes, availableBytes);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    OP_LOGD("aclblasChemm", "tiling: m=%u n=%u k=%u aivCores=%u aicCores=%u ws=%zu/%zu",
        uM, uN, uK, tiling.aivCoreNum, tiling.aicCoreNum, requiredBytes, availableBytes);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchChemmKernel(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B,
    int64_t ldb, const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    ChemmMmadTiling tiling = {};
    aclblasStatus_t status = ConfigureChemmTiling(
        h, side, uplo, m, n, alpha, lda, ldb, beta, ldc, tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    void* workspace = GetEffectiveWorkspace(h);

    OP_LOGI("aclblasChemm", "launching: side=%d uplo=%d m=%ld n=%ld",
            static_cast<int>(side), static_cast<int>(uplo),
            static_cast<long>(m), static_cast<long>(n));

    chemm_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)),
        reinterpret_cast<GM_ADDR>(C),
        reinterpret_cast<GM_ADDR>(workspace),
        tiling, tiling.aivCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B,
    int64_t ldb, const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    bool quickReturn = false;
    aclblasStatus_t ret = ValidateChemmParams(handle, side, uplo, m, n, &quickReturn);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (quickReturn) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    ret = ValidateChemmDataParams(side, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return LaunchChemmKernel(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
