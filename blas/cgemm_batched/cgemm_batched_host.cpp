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

constexpr int ONE = 1;
constexpr int TWO = 2;
constexpr uint32_t GATHER_OFFSETS_SIZE = 1024;

struct CgemmBatchedTilingData {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t batch;
    int64_t subBatch;
    int64_t miniBatch;
};

int aclblasCgemmBatched(aclblasHandle handle,
                        const int64_t m, const int64_t k, const int64_t n,
                        const int64_t batchCount,
                        const float *A, const int64_t lda,
                        const float *B, const int64_t ldb,
                        float *C, const int64_t ldc,
                        void *stream)
{
    constexpr int64_t UB_SIZE = 192 * 1024;
    constexpr int64_t GATHER_OFFSET_SIZE = 1024;
    constexpr int64_t NUM_TWO = 2;
    constexpr int64_t COMPLEX_SIZE = 8;

    int64_t maxCubeCores = 20;
    int64_t useCubeCores = batchCount < maxCubeCores ? batchCount : maxCubeCores;
    
    int64_t subBatch = (batchCount + useCubeCores - 1) / useCubeCores;
    
    int64_t rightSize = k * n;
    int64_t remainUb = UB_SIZE - GATHER_OFFSET_SIZE * sizeof(uint32_t);
    int64_t miniBatch = remainUb / NUM_TWO / NUM_TWO / (rightSize * COMPLEX_SIZE);
    if (miniBatch < 1) miniBatch = 1;
    
    CgemmBatchedTilingData tiling;
    tiling.m = m;
    tiling.k = k;
    tiling.n = n;
    tiling.batch = batchCount;
    tiling.subBatch = subBatch;
    tiling.miniBatch = miniBatch;

    size_t aSize = batchCount * m * k * 2 * sizeof(float);
    size_t bSize = batchCount * k * n * 2 * sizeof(float);
    size_t cSize = batchCount * m * n * 2 * sizeof(float);
    size_t gatherSize = 1024 * sizeof(uint32_t);
    size_t workspaceSize = NUM_TWO * NUM_TWO * miniBatch * rightSize * COMPLEX_SIZE * useCubeCores;

    uint8_t* AHost = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    uint8_t* BHost = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* CHost = reinterpret_cast<uint8_t*>(C);

    uint8_t* ADevice = nullptr;
    uint8_t* BDevice = nullptr;
    uint8_t* CDevice = nullptr;
    uint8_t* gatherDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&BDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&CDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(CgemmBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(ADevice, aSize, AHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(BDevice, bSize, BHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(CDevice, cSize, CHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CgemmBatchedTilingData), &tiling, sizeof(CgemmBatchedTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    uint32_t* gatherData = new uint32_t[GATHER_OFFSETS_SIZE];
    for (size_t i = 0; i < GATHER_OFFSETS_SIZE; i++) {
        if (i % TWO == 0) {
            gatherData[i] = sizeof(float) * (i + 1);
        } else {
            gatherData[i] = sizeof(float) * (i - 1);
        }
    }
    aclrtMemcpy(gatherDevice, gatherSize, gatherData, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE);

    cgemm_batched_kernel_do(ADevice, BDevice, gatherDevice, CDevice, workspaceDevice, tilingDevice, useCubeCores, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(CHost, cSize, CDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(ADevice);
    aclrtFree(BDevice);
    aclrtFree(CDevice);
    aclrtFree(gatherDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    delete[] gatherData;

    return ACL_SUCCESS;
}