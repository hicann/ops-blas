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

constexpr int64_t PAD_NUM = 128;
constexpr int64_t M0 = 128;
constexpr int64_t N0 = 128;
constexpr int64_t K0 = 128;

enum KernelTransType {
    KERNEL_OP_N = 0,
    KERNEL_OP_T = 1,
    KERNEL_OP_C = 2
};

struct CgemmTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t transA;
    int64_t transB;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldaPad;
    int64_t ldbPad;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

static int64_t ConvertTransType(aclblasOperation trans)
{
    if (trans == ACLBLAS_OP_N) return KERNEL_OP_N;
    if (trans == ACLBLAS_OP_T) return KERNEL_OP_T;
    if (trans == ACLBLAS_OP_C) return KERNEL_OP_C;
    return KERNEL_OP_N;
}

int aclblasCgemm(aclblasHandle handle,
                 aclblasOperation transA, aclblasOperation transB,
                 const int64_t m, const int64_t n, const int64_t k,
                 const std::complex<float> *alpha,
                 const float *A, const int64_t lda,
                 const float *B, const int64_t ldb,
                 const std::complex<float> *beta,
                 float *C, const int64_t ldc,
                 void *stream)
{
    int64_t kernelTransA = ConvertTransType(transA);
    int64_t kernelTransB = ConvertTransType(transB);
    
    int64_t ldaPad = (transA == ACLBLAS_OP_N) ? ((m + PAD_NUM - 1) / PAD_NUM * PAD_NUM)
                                              : ((k + PAD_NUM - 1) / PAD_NUM * PAD_NUM);
    int64_t ldbPad = (transB == ACLBLAS_OP_N) ? ((k + PAD_NUM - 1) / PAD_NUM * PAD_NUM)
                                              : ((n + PAD_NUM - 1) / PAD_NUM * PAD_NUM);
    
    CgemmTilingData tiling;
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.transA = kernelTransA;
    tiling.transB = kernelTransB;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.ldaPad = ldaPad;
    tiling.ldbPad = ldbPad;
    tiling.alphaReal = alpha->real();
    tiling.alphaImag = alpha->imag();
    tiling.betaReal = beta->real();
    tiling.betaImag = beta->imag();
    
    int64_t aRows = (transA == ACLBLAS_OP_N) ? m : k;
    int64_t aCols = (transA == ACLBLAS_OP_N) ? k : m;
    int64_t bRows = (transB == ACLBLAS_OP_N) ? k : n;
    int64_t bCols = (transB == ACLBLAS_OP_N) ? n : k;
    
    size_t aSize = aRows * aCols * 2 * sizeof(float);
    size_t bSize = bRows * bCols * 2 * sizeof(float);
    size_t cSize = m * n * 2 * sizeof(float);
    
    int64_t aPadSizeCount = (transA == ACLBLAS_OP_N) ?
        (ldaPad * (k + K0 - 1) / K0 * K0) :
        (ldaPad * (m + M0 - 1) / M0 * M0);
    int64_t bPadSizeCount = (transB == ACLBLAS_OP_N) ?
        (ldbPad * (n + N0 - 1) / N0 * N0) :
        (ldbPad * (k + K0 - 1) / K0 * K0);
    int64_t cResultSizeCount = ldc * n;
    
    size_t aPadSize = aPadSizeCount * sizeof(float);
    size_t bPadSize = bPadSizeCount * sizeof(float);
    size_t cResultSize = cResultSizeCount * sizeof(float);
    
    size_t workspaceSize = 16 * 1024 * 1024;
    
    uint8_t* AHost = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    uint8_t* BHost = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* CHost = reinterpret_cast<uint8_t*>(C);
    
    uint8_t* ADevice = nullptr;
    uint8_t* BDevice = nullptr;
    uint8_t* CDevice = nullptr;
    uint8_t* A_rDevice = nullptr;
    uint8_t* A_iDevice = nullptr;
    uint8_t* B_rDevice = nullptr;
    uint8_t* B_iDevice = nullptr;
    uint8_t* C_rrDevice = nullptr;
    uint8_t* C_riDevice = nullptr;
    uint8_t* C_irDevice = nullptr;
    uint8_t* C_iiDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    
    aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&BDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&CDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&A_rDevice, aPadSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&A_iDevice, aPadSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&B_rDevice, bPadSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&B_iDevice, bPadSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&C_rrDevice, cResultSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&C_riDevice, cResultSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&C_irDevice, cResultSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&C_iiDevice, cResultSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(CgemmTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtMemcpy(ADevice, aSize, AHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(BDevice, bSize, BHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(CDevice, cSize, CHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CgemmTilingData), &tiling, sizeof(CgemmTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    
    uint32_t numBlocks = 8;
    
    cgemm_kernel_do(ADevice, BDevice, A_rDevice, A_iDevice,
                    B_rDevice, B_iDevice, C_rrDevice, C_riDevice,
                    C_irDevice, C_iiDevice, CDevice, workspaceDevice,
                    tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtMemcpy(CHost, cSize, CDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclrtFree(ADevice);
    aclrtFree(BDevice);
    aclrtFree(CDevice);
    aclrtFree(A_rDevice);
    aclrtFree(A_iDevice);
    aclrtFree(B_rDevice);
    aclrtFree(B_iDevice);
    aclrtFree(C_rrDevice);
    aclrtFree(C_riDevice);
    aclrtFree(C_irDevice);
    aclrtFree(C_iiDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);
    
    return ACL_SUCCESS;
}