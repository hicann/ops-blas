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

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t WORKSPACE_SIZE = 1024;

struct SsyrTilingData {
    uint32_t uplo;
    uint32_t n;
    float alpha;
    uint32_t coreNum;
};

int aclblasSsyr(aclblasHandle handle,
                aclblasFillMode uplo,
                const int64_t n,
                const float alpha,
                const float *x, const int64_t incx,
                float *A, const int64_t lda,
                void *stream)
{
    uint32_t vecCoreNum = DEFAULT_VECTOR_NUM;

    SsyrTilingData tiling;
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    tiling.n = n;
    tiling.alpha = alpha;
    tiling.coreNum = vecCoreNum;

    size_t matSize = lda * n * sizeof(float);
    size_t xSize = n * incx * sizeof(float);

    uint8_t* AHost = reinterpret_cast<uint8_t*>(A);
    uint8_t* xHost = reinterpret_cast<uint8_t*>(const_cast<float*>(x));

    uint8_t* ADevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc((void**)&ADevice, matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(SsyrTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(ADevice, matSize, AHost, matSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, xSize, xHost, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(SsyrTilingData), &tiling, sizeof(SsyrTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    ssyr_kernel_do(xDevice, ADevice, workSpaceDevice, tilingDevice, vecCoreNum, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(AHost, matSize, ADevice, matSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(ADevice);
    aclrtFree(xDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}