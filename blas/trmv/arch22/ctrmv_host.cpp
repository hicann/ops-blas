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
 * \file ctrmv_host.cpp
 * \brief Host side implementation for ctrmv operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void ctrmv_kernel_do(uint8_t* gm_A, uint8_t* gm_X, uint8_t* gm_uplo,
                     uint8_t* gm_wksp, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

constexpr int64_t BASIC_DATA_PROC_CNT = 64;
constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr int64_t N0 = 64;

// Tiling data structure - consistent with sip CtrmvTilingData
struct CtrmvTilingData {
    int64_t mode;
    int64_t trans;
    int64_t diag;
    int64_t n;
    int64_t lda;
    int64_t incx;
    int64_t n0;
};

// Tiling calculation - migrated from sip ctrmv_tiling.cpp
CtrmvTilingData CalCtrmvTilingData(int64_t mode, int64_t trans, int64_t diag, int64_t n, int64_t lda, int64_t incx)
{
    CtrmvTilingData tilingData;
    memset(&tilingData, 0, sizeof(CtrmvTilingData));

    tilingData.mode = mode;
    tilingData.trans = trans;
    tilingData.diag = diag;
    tilingData.n = n;
    tilingData.lda = lda;
    tilingData.incx = incx;
    tilingData.n0 = BASIC_DATA_PROC_CNT;

    return tilingData;
}

// Plan: Create uplo mask matrix - migrated from sip BlasCtrmvPlan.cpp
float* CreateCtrmvUploMatrix(int64_t uplo)
{
    int64_t blockSize = N0 * N0;
    float* uploMatrixData = new float[blockSize];

    float ele = (uplo == 0) ? 0 : 1; // 0 = LOWER, 1 = UPPER

    for (int64_t i = 0; i < N0; ++i) {
        for (int64_t j = 0; j < N0; ++j) {
            if (j < i) {
                *(uploMatrixData + i * N0 + j) = ele;
            } else if (j == i) {
                *(uploMatrixData + i * N0 + j) = 1;
            } else {
                *(uploMatrixData + i * N0 + j) = 1 - ele;
            }
        }
    }

    return uploMatrixData;
}

// Calculate block dim - migrated from sip ctrmv_tiling.cpp
uint32_t CalCtrmvBlockDim(int64_t n, uint32_t coreNum)
{
    int64_t nDupNum = (n - 1) / BASIC_DATA_PROC_CNT + 1;
    int64_t groupDim = nDupNum * nDupNum;

    groupDim = groupDim < coreNum ? groupDim : coreNum;
    if (groupDim == 0) {
        groupDim = 1;
    }
    return static_cast<uint32_t>(groupDim);
}

aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    uint8_t* A, int64_t lda, uint8_t* x, int64_t incx)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    int64_t uploLocal = (uplo == ACLBLAS_LOWER) ? 0 : 1;
    int64_t transLocal = -1;
    if (trans == ACLBLAS_OP_N) {
        transLocal = 0;
    } else if (trans == ACLBLAS_OP_T) {
        transLocal = 1;
    } else if (trans == ACLBLAS_OP_C) {
        transLocal = 2;
    }
    int64_t diagLocal = (diag == ACLBLAS_NON_UNIT) ? 0 : 1;

    CtrmvTilingData tiling = CalCtrmvTilingData(uploLocal, transLocal, diagLocal, n, lda, incx);

    uint32_t coreNum = 8;
    uint32_t numBlocks = CalCtrmvBlockDim(n, coreNum);

    float* uploMatrixData = CreateCtrmvUploMatrix(uploLocal);

    size_t uploByteSize = N0 * N0 * sizeof(float);
    int64_t mTiles = (n + BASIC_DATA_PROC_CNT - 1) / BASIC_DATA_PROC_CNT;
    size_t workspaceSize = mTiles * BASIC_DATA_PROC_CNT * ELEMENTS_EACH_COMPLEX64 * sizeof(float);
    if (workspaceSize < 1024) {
        workspaceSize = 1024;
    }

    uint8_t* uploHost = reinterpret_cast<uint8_t*>(uploMatrixData);

    uint8_t* uploDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&uploDevice, uploByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); delete[] uploMatrixData;
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(uploDevice);
        delete[] uploMatrixData; return ACLBLAS_STATUS_ALLOC_FAILED);

    size_t tilingSize = (sizeof(CtrmvTilingData) + 31) / 32 * 32;
    aclRet = aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
        aclrtFree(uploDevice); delete[] uploMatrixData; return ACLBLAS_STATUS_ALLOC_FAILED);

    std::vector<uint8_t> workspaceHost(workspaceSize, 0);
    aclRet =
        aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost.data(), workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(uploDevice); delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet = aclrtMemcpy(uploDevice, uploByteSize, uploHost, uploByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(uploDevice); delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CtrmvTilingData), &tiling, sizeof(CtrmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(uploDevice); delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    ctrmv_kernel_do(A, x, uploDevice, workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); aclrtFree(uploDevice); delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(uploDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    delete[] uploMatrixData;

    return ACLBLAS_STATUS_SUCCESS;
}
