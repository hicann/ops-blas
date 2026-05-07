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
#include <complex>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../utils/aclblas_kernel_do.h"

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

int aclblasCgemv(aclblasHandle handle,
                  aclblasOperation trans,
                  const int64_t m, const int64_t n,
                  const std::complex<float> &alpha,
                  const std::complex<float> *A, const int64_t lda,
                  const std::complex<float> *x, const int64_t incx,
                  const std::complex<float> &beta,
                  std::complex<float> *y, const int64_t incy,
                  void *stream)
{
    uint32_t numBlocks = 8;

    int64_t actualM = (trans == ACLBLAS_OP_N) ? m : n;
    int64_t actualN = (trans == ACLBLAS_OP_N) ? n : m;

    size_t matSize = m * n * 2 * sizeof(float);
    size_t xSize = actualN * 2 * sizeof(float);
    size_t ySize = actualM * 2 * sizeof(float);

    CgemvTilingData tiling;
    tiling.transA = (int64_t)trans;
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.sectionDim = 4;
    tiling.alphaReal = alpha.real();
    tiling.alphaImag = alpha.imag();
    tiling.betaReal = beta.real();
    tiling.betaImag = beta.imag();

    uint32_t* mask = CreateCgemvMask(m);
    size_t maskSize = MASK_OFFSET_BASE * 2 * sizeof(uint32_t);

    size_t workSpaceSize = WORKSPACE_SIZE;

    uint8_t* AHost = reinterpret_cast<uint8_t*>(const_cast<std::complex<float>*>(A));
    uint8_t* xHost = reinterpret_cast<uint8_t*>(const_cast<std::complex<float>*>(x));
    uint8_t* yHost = reinterpret_cast<uint8_t*>(y);
    uint8_t* maskHost = reinterpret_cast<uint8_t*>(mask);

    uint8_t* ADevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* yInDevice = nullptr;
    uint8_t* maskDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclrtMalloc((void**)&ADevice, matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yInDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(CgemvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(ADevice, matSize, AHost, matSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, xSize, xHost, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, ySize, yHost, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yInDevice, ySize, yHost, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CgemvTilingData), &tiling, sizeof(CgemvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    if (trans == ACLBLAS_OP_N) {
        cgemv_no_trans_kernel_do(ADevice, xDevice, yInDevice, maskDevice, yDevice,
                                 workSpaceDevice, tilingDevice, numBlocks, stream);
    } else {
        cgemv_do_trans_kernel_do(ADevice, xDevice, yInDevice, maskDevice, yDevice,
                                 workSpaceDevice, tilingDevice, numBlocks, stream);
    }
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(yHost, ySize, yDevice, ySize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(ADevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(yInDevice);
    aclrtFree(maskDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    delete[] mask;

    return ACL_SUCCESS;
}