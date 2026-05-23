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
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

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

aclblasStatus_t aclblasCgemmBatched(aclblasHandle handle,
                                     aclblasOperation transa, aclblasOperation transb,
                                     const int64_t m, const int64_t n, const int64_t k,
                                     const std::complex<float> &alpha,
                                     uint8_t *A, const int64_t lda,
                                     uint8_t *B, const int64_t ldb,
                                     const std::complex<float> &beta,
                                     uint8_t *C, const int64_t ldc,
                                     const int64_t batchCount)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;
    
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

    size_t gatherSize = 1024 * sizeof(uint32_t);
    size_t workspaceSize = NUM_TWO * NUM_TWO * miniBatch * rightSize * COMPLEX_SIZE * useCubeCores;

    uint8_t* gatherDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(gatherDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CgemmBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice); aclrtFree(gatherDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(tilingDevice, sizeof(CgemmBatchedTilingData), &tiling, sizeof(CgemmBatchedTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workspaceDevice); aclrtFree(gatherDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    uint32_t* gatherData = new uint32_t[GATHER_OFFSETS_SIZE];
    for (size_t i = 0; i < GATHER_OFFSETS_SIZE; i++) {
        if (i % TWO == 0) {
            gatherData[i] = sizeof(float) * (i + 1);
        } else {
            gatherData[i] = sizeof(float) * (i - 1);
        }
    }
    aclRet = aclrtMemcpy(gatherDevice, gatherSize, gatherData, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workspaceDevice); aclrtFree(gatherDevice); delete[] gatherData; return ACLBLAS_STATUS_INTERNAL_ERROR);

    cgemm_batched_kernel_do(A, B, gatherDevice, C, workspaceDevice, tilingDevice, useCubeCores, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workspaceDevice); aclrtFree(gatherDevice); delete[] gatherData; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(gatherDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    delete[] gatherData;

    return ACLBLAS_STATUS_SUCCESS;
}