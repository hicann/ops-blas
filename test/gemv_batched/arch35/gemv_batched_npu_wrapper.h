/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of License.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

// ---- dtype 1 (S): float in, float out ----
inline aclblasStatus_t aclblasGemvBatchedS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasSgemvBatched(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(batchCount) * allocLda * std::max(1, n) * sizeof(float);
    const size_t xBytes = static_cast<size_t>(batchCount) * ((xCount - 1) * std::abs(incx) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(batchCount) * ((yCount - 1) * std::abs(incy) + 1) * sizeof(float);

    DeviceBuffer dA(aBytes), dX(xBytes), dY(yBytes);
    dA.copyFromHost(a, aBytes);
    dX.copyFromHost(x, xBytes);
    dY.copyFromHost(y, yBytes);
    aclblasStatus_t ret = aclblasSgemvBatched(
        handle, trans, m, n, alpha, static_cast<const float*>(dA.ptr()), lda,
        static_cast<const float*>(dX.ptr()), incx, beta,
        static_cast<float*>(dY.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    dY.copyToHost(y, yBytes);
    return ret;
}

// ---- dtype 0 (HSH): uint16_t in, uint16_t out ----
inline aclblasStatus_t aclblasGemvBatchedHSH_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasHSHgemvBatched(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)  return ACLBLAS_STATUS_INVALID_VALUE;
    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(batchCount) * allocLda * std::max(1, n) * sizeof(uint16_t);
    const size_t xBytes = static_cast<size_t>(batchCount) * ((xCount - 1) * std::abs(incx) + 1) * sizeof(uint16_t);
    const size_t yBytes = static_cast<size_t>(batchCount) * ((yCount - 1) * std::abs(incy) + 1) * sizeof(uint16_t);

    DeviceBuffer dA(aBytes), dX(xBytes), dY(yBytes);
    dA.copyFromHost(a, aBytes);
    dX.copyFromHost(x, xBytes);
    dY.copyFromHost(y, yBytes);
    aclblasStatus_t ret = aclblasHSHgemvBatched(
        handle, trans, m, n, alpha, static_cast<const uint16_t*>(dA.ptr()), lda,
        static_cast<const uint16_t*>(dX.ptr()), incx, beta,
        static_cast<uint16_t*>(dY.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    dY.copyToHost(y, yBytes);
    return ret;
}

// ---- dtype 2 (HSS): uint16_t in, float out ----
inline aclblasStatus_t aclblasGemvBatchedHSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasHSSgemvBatched(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)  return ACLBLAS_STATUS_INVALID_VALUE;
    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(batchCount) * allocLda * std::max(1, n) * sizeof(uint16_t);
    const size_t xBytes = static_cast<size_t>(batchCount) * ((xCount - 1) * std::abs(incx) + 1) * sizeof(uint16_t);
    const size_t yBytes = static_cast<size_t>(batchCount) * ((yCount - 1) * std::abs(incy) + 1) * sizeof(float);

    DeviceBuffer dA(aBytes), dX(xBytes), dY(yBytes);
    dA.copyFromHost(a, aBytes);
    dX.copyFromHost(x, xBytes);
    dY.copyFromHost(y, yBytes);
    aclblasStatus_t ret = aclblasHSSgemvBatched(
        handle, trans, m, n, alpha, static_cast<const uint16_t*>(dA.ptr()), lda,
        static_cast<const uint16_t*>(dX.ptr()), incx, beta,
        static_cast<float*>(dY.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    dY.copyToHost(y, yBytes);
    return ret;
}

// ---- dtype 3 (TST): uint16_t in, uint16_t out (bf16) ----
inline aclblasStatus_t aclblasGemvBatchedTST_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasTSTgemvBatched(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)  return ACLBLAS_STATUS_INVALID_VALUE;
    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(batchCount) * allocLda * std::max(1, n) * sizeof(uint16_t);
    const size_t xBytes = static_cast<size_t>(batchCount) * ((xCount - 1) * std::abs(incx) + 1) * sizeof(uint16_t);
    const size_t yBytes = static_cast<size_t>(batchCount) * ((yCount - 1) * std::abs(incy) + 1) * sizeof(uint16_t);

    DeviceBuffer dA(aBytes), dX(xBytes), dY(yBytes);
    dA.copyFromHost(a, aBytes);
    dX.copyFromHost(x, xBytes);
    dY.copyFromHost(y, yBytes);
    aclblasStatus_t ret = aclblasTSTgemvBatched(
        handle, trans, m, n, alpha, static_cast<const uint16_t*>(dA.ptr()), lda,
        static_cast<const uint16_t*>(dX.ptr()), incx, beta,
        static_cast<uint16_t*>(dY.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    dY.copyToHost(y, yBytes);
    return ret;
}

// ---- dtype 4 (TSS): uint16_t in, float out (bf16) ----
inline aclblasStatus_t aclblasGemvBatchedTSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasTSSgemvBatched(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)  return ACLBLAS_STATUS_INVALID_VALUE;
    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(batchCount) * allocLda * std::max(1, n) * sizeof(uint16_t);
    const size_t xBytes = static_cast<size_t>(batchCount) * ((xCount - 1) * std::abs(incx) + 1) * sizeof(uint16_t);
    const size_t yBytes = static_cast<size_t>(batchCount) * ((yCount - 1) * std::abs(incy) + 1) * sizeof(float);

    DeviceBuffer dA(aBytes), dX(xBytes), dY(yBytes);
    dA.copyFromHost(a, aBytes);
    dX.copyFromHost(x, xBytes);
    dY.copyFromHost(y, yBytes);
    aclblasStatus_t ret = aclblasTSSgemvBatched(
        handle, trans, m, n, alpha, static_cast<const uint16_t*>(dA.ptr()), lda,
        static_cast<const uint16_t*>(dX.ptr()), incx, beta,
        static_cast<float*>(dY.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    dY.copyToHost(y, yBytes);
    return ret;
}

