/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gbmv_host.cpp
 * \brief Host-side implementation for aclblasSgbmv (banded matrix-vector multiplication).
 *
 * Supports trans='N', 'T', 'C' (C treated as T for real types) and FP32/FP64 data types.
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "gbmv_tiling_data.h"

constexpr uint32_t GBMV_MAX_CORE_NUM = 50;

template <typename T>
static GbmvTilingData<T> CalGbmvTilingData(
    uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda, int32_t trans, T alpha, T beta)
{
    GbmvTilingData<T> tilingData{};
    tilingData.m = m;
    tilingData.n = n;
    tilingData.kl = kl;
    tilingData.ku = ku;
    tilingData.lda = lda;
    tilingData.trans = trans;
    tilingData.alpha = alpha;
    tilingData.beta = beta;

    uint32_t availableCoreNum = (n < GBMV_MAX_CORE_NUM) ? n : GBMV_MAX_CORE_NUM;
    if (availableCoreNum == 0) {
        availableCoreNum = 1;
    }
    tilingData.useCoreNum = availableCoreNum;

    if (trans == 0) { // TRANS_N
        tilingData.maxSegLen = (kl + ku + 1 < m) ? (kl + ku + 1) : m;
    } else {
        tilingData.maxSegLen = (kl + ku + 1 < n) ? (kl + ku + 1) : n;
    }

    return tilingData;
}

static aclblasStatus_t ValidateGbmvHandleAndTrans(aclblasHandle_t handle, aclblasOperation_t trans)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGbmvValueParams(
    int m, int n, int kl, int ku, int lda, int incx, int incy, const void* A, const void* x, const void* y,
    const void* alpha, const void* beta)
{
    if (m < 0 || n < 0 || kl < 0 || ku < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < kl + ku + 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr && m > 0 && n > 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static int32_t MapTrans(aclblasOperation_t trans)
{
    if (trans == ACLBLAS_OP_N) {
        return 0;
    }
    return 1;
}

template <typename T>
static void GatherStrided(const T* src, int64_t count, int64_t stride, std::vector<T>& contigBuf, const T*& outPtr)
{
    if (stride == 1) {
        outPtr = src;
    } else {
        contigBuf.resize(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; i++) {
            contigBuf[i] = src[i * stride];
        }
        outPtr = contigBuf.data();
    }
}

template <typename T>
static void ScatterStrided(T* dst, const std::vector<T>& src, int64_t count, int64_t stride)
{
    for (int64_t i = 0; i < count; i++) {
        dst[i * stride] = src[i];
    }
}

struct GbmvDeviceBuffers {
    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* zDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    void FreeAll()
    {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(zDevice);
        aclrtFree(tilingDevice);
    }
};

static aclblasStatus_t AllocGbmvDeviceBuffers(
    GbmvDeviceBuffers& buf, size_t matrixByteSize, size_t xContigByteSize, size_t yContigByteSize,
    size_t tilingByteSize)
{
    aclError ret;
    ret = aclrtMalloc(reinterpret_cast<void**>(&buf.aDevice), matrixByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    ret = aclrtMalloc(reinterpret_cast<void**>(&buf.xDevice), xContigByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        aclrtFree(buf.aDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMalloc(reinterpret_cast<void**>(&buf.yDevice), yContigByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        aclrtFree(buf.aDevice);
        aclrtFree(buf.xDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMalloc(reinterpret_cast<void**>(&buf.zDevice), yContigByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        aclrtFree(buf.aDevice);
        aclrtFree(buf.xDevice);
        aclrtFree(buf.yDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMalloc(reinterpret_cast<void**>(&buf.tilingDevice), tilingByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        aclrtFree(buf.aDevice);
        aclrtFree(buf.xDevice);
        aclrtFree(buf.yDevice);
        aclrtFree(buf.zDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void DispatchGbmvAndCopyResult(
    const GbmvDeviceBuffers& buf, const void* A, const void* xForDevice, const void* yForDevice, size_t matrixByteSize,
    size_t xContigByteSize, size_t yContigByteSize, const void* tiling, size_t tilingByteSize, uint32_t numBlocks,
    aclrtStream stream, float* y, int64_t incyPara, uint32_t yCount)
{
    aclrtMemcpy(buf.aDevice, matrixByteSize, A, matrixByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.xDevice, xContigByteSize, xForDevice, xContigByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.yDevice, yContigByteSize, yForDevice, yContigByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.tilingDevice, tilingByteSize, tiling, tilingByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(buf.zDevice, yContigByteSize, 0, yContigByteSize);

    gbmv_kernel_do(buf.aDevice, buf.xDevice, buf.yDevice, buf.zDevice, nullptr, buf.tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    if (incyPara != 1) {
        std::vector<float> zHost(yCount);
        aclrtMemcpy(zHost.data(), yContigByteSize, buf.zDevice, yContigByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        ScatterStrided(y, zHost, yCount, incyPara);
    } else {
        aclrtMemcpy(y, yContigByteSize, buf.zDevice, yContigByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
}

// ===========================================================================
// aclblasSgbmv �?FP32 banded matrix-vector multiply
// ===========================================================================
aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy)
{
    aclblasStatus_t ret = ValidateGbmvHandleAndTrans(handle, trans);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;

    ret = ValidateGbmvValueParams(m, n, kl, ku, lda, incx, incy, A, x, y, alpha, beta);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    uint32_t uiM = static_cast<uint32_t>(m);
    uint32_t uiN = static_cast<uint32_t>(n);
    int64_t absIncx = (incx < 0) ? -incx : incx;
    int64_t absIncy = (incy < 0) ? -incy : incy;

    bool isTransN = (trans == ACLBLAS_OP_N);
    uint32_t xCount = isTransN ? uiN : uiM;
    uint32_t yCount = isTransN ? uiM : uiN;

    size_t matrixByteSize = static_cast<size_t>(uiN) * static_cast<uint32_t>(lda) * sizeof(float);
    size_t xContigByteSize = static_cast<size_t>(xCount) * sizeof(float);
    size_t yContigByteSize = static_cast<size_t>(yCount) * sizeof(float);

    std::vector<float> xContig;
    const float* xForDevice = x;
    GatherStrided(x, xCount, incx, xContig, xForDevice);

    std::vector<float> yContig;
    const float* yForDevice = y;
    GatherStrided(y, yCount, incy, yContig, yForDevice);

    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);

    int32_t transInt = MapTrans(trans);
    GbmvTilingData<float> tiling = CalGbmvTilingData<float>(
        uiM, uiN, static_cast<uint32_t>(kl), static_cast<uint32_t>(ku), static_cast<uint32_t>(lda), transInt, *alpha,
        *beta);

    GbmvDeviceBuffers buf;
    ret = AllocGbmvDeviceBuffers(buf, matrixByteSize, xContigByteSize, yContigByteSize, sizeof(GbmvTilingData<float>));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;

    DispatchGbmvAndCopyResult(
        buf, A, xForDevice, yForDevice, matrixByteSize, xContigByteSize, yContigByteSize, &tiling,
        sizeof(GbmvTilingData<float>), tiling.useCoreNum, stream, y, incy, yCount);

    buf.FreeAll();
    return ACLBLAS_STATUS_SUCCESS;
}
