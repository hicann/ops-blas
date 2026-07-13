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
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgemv_do_trans_kernel_do(uint8_t* d_A, uint8_t* d_x, uint8_t* d_y_in, uint8_t* maskBuf, uint8_t* d_y,
                              uint8_t* workSpace, uint8_t* tilingGm,
                              uint32_t numBlocks, void *stream);
void cgemv_no_trans_kernel_do(uint8_t* d_A, uint8_t* d_x, uint8_t* d_y_in, uint8_t* maskBuf, uint8_t* d_y,
                              uint8_t* workSpace, uint8_t* tilingGm,
                              uint32_t numBlocks, void *stream);

constexpr uint32_t MAX_CORE_CNT = 40;
constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MASK_OFFSET_BASE = 1024;

struct CgemvTilingData {
    int64_t transA;
    int64_t m;
    int64_t n;
    int64_t lda;
    int64_t incx;
    int64_t incy;
    int64_t sectionDim;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

static uint32_t* CreateCgemvMask(int64_t m)
{
    uint32_t realOffset = 0;
    uint32_t imagOffset = MASK_OFFSET_BASE;

    uint32_t maskSize = imagOffset * 2;
    uint32_t* maskData = new uint32_t[maskSize];

    int k = 0;
    for (uint32_t i = 0; i < maskSize / 2; i++) {
        maskData[k++] = (realOffset + i) * sizeof(float);
        maskData[k++] = (imagOffset + i) * sizeof(float);
    }

    return maskData;
}

aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, const int64_t m, const int64_t n, const aclblasComplex alpha,
    aclblasComplex* A, const int64_t lda, aclblasComplex* x, const int64_t incx, const aclblasComplex beta,
    aclblasComplex* y, const int64_t incy)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    CgemvTilingData tiling;
    tiling.transA = (int64_t)trans;
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.sectionDim = 4;
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.betaReal = beta.real;
    tiling.betaImag = beta.imag;

    uint32_t* mask = CreateCgemvMask(m);
    size_t maskSize = MASK_OFFSET_BASE * 2 * sizeof(uint32_t);
    size_t workSpaceSize = WORKSPACE_SIZE;

    uint8_t* maskDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); delete[] mask;
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(maskDevice);
        delete[] mask; return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CgemvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
        aclrtFree(maskDevice); delete[] mask; return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(maskDevice, maskSize, mask, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); delete[] mask; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CgemvTilingData), &tiling, sizeof(CgemvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); delete[] mask; return ACLBLAS_STATUS_INTERNAL_ERROR);

    if (trans == ACLBLAS_OP_N) {
        cgemv_no_trans_kernel_do(reinterpret_cast<uint8_t*>(A), reinterpret_cast<uint8_t*>(x),
                                 reinterpret_cast<uint8_t*>(y), maskDevice, reinterpret_cast<uint8_t*>(y),
                                 workSpaceDevice, tilingDevice, numBlocks, useStream);
    } else {
        cgemv_do_trans_kernel_do(reinterpret_cast<uint8_t*>(A), reinterpret_cast<uint8_t*>(x),
                                 reinterpret_cast<uint8_t*>(y), maskDevice, reinterpret_cast<uint8_t*>(y),
                                 workSpaceDevice, tilingDevice, numBlocks, useStream);
    }
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); delete[] mask; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(maskDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    delete[] mask;

    return ACLBLAS_STATUS_SUCCESS;
}