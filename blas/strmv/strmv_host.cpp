/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
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

constexpr uint32_t CORE_SPLIT_NUM = 128;
constexpr uint32_t WORKSPACE_SIZE = 128 * 1024 * sizeof(float);

struct StrmvTilingData {
    uint32_t matSizeN;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    uint32_t lda;
    uint32_t incx;
    uint32_t m0;
};

static float* CreateUploMatrix(int64_t uplo)
{
    constexpr int64_t M0 = 128;
    float* uploData = new float[M0 * M0];
    for (int64_t i = 0; i < M0; i++) {
        for (int64_t j = 0; j < M0; j++) {
            if (uplo == ACLBLAS_UPPER) {
                uploData[i + j * M0] = (j >= i) ? 1.0f : 0.0f;
            } else {
                uploData[i + j * M0] = (j <= i) ? 1.0f : 0.0f;
            }
        }
    }
    return uploData;
}

int aclblasStrmv(aclblasHandle handle,
                 aclblasFillMode uplo,
                 aclblasOperation trans,
                 aclblasDiagType diag,
                 const int64_t n,
                 const float *A, const int64_t lda,
                 float *x, const int64_t incx,
                 void *stream)
{
    uint32_t m0 = CORE_SPLIT_NUM;
    uint32_t m0TileNumOfM = (n + m0 - 1) / m0;
    uint32_t numBlocks = m0TileNumOfM;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    StrmvTilingData tiling;
    tiling.matSizeN = n;
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    tiling.trans = (trans == ACLBLAS_OP_N) ? 0 : 1;
    tiling.diag = (diag == ACLBLAS_UNIT) ? 1 : 0;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.m0 = m0;

    float* uploMatrix = CreateUploMatrix(uplo);

    size_t matSize = n * n * sizeof(float);
    size_t xSize = n * incx * sizeof(float);
    size_t wkspSize = n * sizeof(float);
    size_t uploSize = 128 * 128 * sizeof(float);

    uint8_t* AHost = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    uint8_t* xHost = reinterpret_cast<uint8_t*>(x);
    uint8_t* uploHost = reinterpret_cast<uint8_t*>(uploMatrix);

    uint8_t* ADevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* uploDevice = nullptr;
    uint8_t* outputDevice = nullptr;
    uint8_t* wkspDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc((void**)&ADevice, matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&uploDevice, uploSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outputDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&wkspDevice, wkspSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(StrmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(ADevice, matSize, AHost, matSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, xSize, xHost, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(uploDevice, uploSize, uploHost, uploSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(StrmvTilingData), &tiling, sizeof(StrmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    strmv_kernel_do(ADevice, xDevice, uploDevice, outputDevice, wkspDevice,
                    workSpaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, xSize, outputDevice, xSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(ADevice);
    aclrtFree(xDevice);
    aclrtFree(uploDevice);
    aclrtFree(outputDevice);
    aclrtFree(wkspDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    delete[] uploMatrix;

    return ACL_SUCCESS;
}