/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../utils/aclblas_kernel_do.h"

constexpr uint32_t CORE_NUM = 8;
constexpr uint32_t WORKSPACE_SIZE = 1024 * 1024;

struct StrmmTilingData {
    uint32_t side;
    uint32_t uplo;
    uint32_t transa;
    uint32_t transb;
    uint32_t diag;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lessFlag;
    float alpha;
};

static uint32_t ConvertSideMode(aclblasSideMode side)
{
    if (side == ACLBLAS_SIDE_LEFT) return 0;
    return 1;
}

static uint32_t ConvertFillMode(aclblasFillMode uplo)
{
    if (uplo == ACLBLAS_UPPER) return 1;
    return 0;
}

static uint32_t ConvertOperation(aclblasOperation trans)
{
    if (trans == ACLBLAS_OP_N) return 0;
    if (trans == ACLBLAS_OP_T) return 1;
    return 2;
}

static uint32_t ConvertDiagType(aclblasDiagType diag)
{
    if (diag == ACLBLAS_UNIT) return 1;
    return 0;
}

int aclblasStrmm(aclblasHandle handle,
                 aclblasSideMode side,
                 aclblasFillMode uplo,
                 aclblasOperation transa,
                 aclblasOperation transb,
                 aclblasDiagType diag,
                 const int64_t m, const int64_t n, const int64_t k,
                 const float alpha,
                 const float *A, const int64_t lda,
                 const float *B, const int64_t ldb,
                 float *C, const int64_t ldc,
                 void *stream)
{
    StrmmTilingData tiling;
    tiling.side = ConvertSideMode(side);
    tiling.uplo = ConvertFillMode(uplo);
    tiling.transa = ConvertOperation(transa);
    tiling.transb = ConvertOperation(transb);
    tiling.diag = ConvertDiagType(diag);
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lessFlag = 0;
    tiling.alpha = alpha;

    int64_t aRows = (transa == ACLBLAS_OP_N) ? m : k;
    int64_t aCols = (transa == ACLBLAS_OP_N) ? k : m;
    int64_t bRows = (transb == ACLBLAS_OP_N) ? k : n;
    int64_t bCols = (transb == ACLBLAS_OP_N) ? n : k;

    size_t aSize = aRows * aCols * sizeof(float);
    size_t bSize = bRows * bCols * sizeof(float);
    size_t cSize = m * n * sizeof(float);

    uint8_t* AHost = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    uint8_t* BHost = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* CHost = reinterpret_cast<uint8_t*>(C);

    uint8_t* ADevice = nullptr;
    uint8_t* BDevice = nullptr;
    uint8_t* CDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&BDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&CDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workspaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(StrmmTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(ADevice, aSize, AHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(BDevice, bSize, BHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(CDevice, cSize, CHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(StrmmTilingData), &tiling, sizeof(StrmmTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    uint32_t numBlocks = CORE_NUM;

    strmm_kernel_do(ADevice, BDevice, CDevice, workspaceDevice,
                    workSpaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(CHost, cSize, CDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(ADevice);
    aclrtFree(BDevice);
    aclrtFree(CDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}