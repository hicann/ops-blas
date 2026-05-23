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

aclblasStatus_t aclblasStrmm(aclblasHandle handle,
                             aclblasSideMode side,
                             aclblasFillMode uplo,
                             aclblasOperation trans,
                             aclblasDiagType diag,
                             const int64_t m, const int64_t n, const float alpha,
                             uint8_t *A, const int64_t lda,
                             uint8_t *B, const int64_t ldb,
                             uint8_t *C, const int64_t ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    StrmmTilingData tiling;
    tiling.side = ConvertSideMode(side);
    tiling.uplo = ConvertFillMode(uplo);
    tiling.transa = ConvertOperation(trans);
    tiling.transb = 0;  // Always N for strmm
    tiling.diag = ConvertDiagType(diag);
    tiling.m = m;
    tiling.n = n;
    tiling.k = (side == ACLBLAS_SIDE_LEFT) ? m : n;  // k depends on side
    tiling.lessFlag = 0;
    tiling.alpha = alpha;

    uint8_t* workspaceDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workspaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(StrmmTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice); aclrtFree(workspaceDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(tilingDevice, sizeof(StrmmTilingData), &tiling, sizeof(StrmmTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workSpaceDevice); aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    uint32_t numBlocks = CORE_NUM;

    strmm_kernel_do(A, B, C, workspaceDevice,
                    workSpaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workSpaceDevice); aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}