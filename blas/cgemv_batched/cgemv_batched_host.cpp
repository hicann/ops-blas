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
 * \file cgemv_batched_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cgemv_batched_plan.h"

using aclblasHandle = void *;

#define GM_ADDR uint8_t*

extern void cgemv_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                                    GM_ADDR workSpace, GM_ADDR tilingGm,
                                    uint32_t numBlocks, void *stream);

constexpr uint32_t MAX_CORE_CNT = 40;
constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CgemvBatchedTilingData {
    uint32_t dtype;
    uint32_t trans;
    uint32_t m;
    uint32_t n;
    uint32_t maxMatNum;
    uint32_t calMatNum[40];
    uint32_t startMatId[40];
};

static CgemvBatchedTilingData CalTilingData(uint32_t batchCount, uint32_t m, uint32_t n,
                                            uint32_t dtype, uint32_t trans, uint32_t vecCoreNum)
{
    CgemvBatchedTilingData tilingData;
    tilingData.dtype = dtype;
    tilingData.trans = trans;
    tilingData.m = m;
    tilingData.n = n;

    // Calculate maxMatNum based on available UB size
    // For simplicity, we set maxMatNum to a reasonable value
    uint32_t maxMatNum = 1;  // Can be adjusted based on actual UB size
    bool isTrans = trans == 0 ? false : true;    // 0: ACLBLAS_OP_N
    bool dataType = (dtype == (uint32_t)aclDataType_t::ACL_C_64F) ? true : false;
    maxMatNum = CalMaxMatNum(isTrans, dataType, static_cast<uint32_t>(m));

    tilingData.maxMatNum = maxMatNum > 0 ? maxMatNum : 1;

    // Initialize arrays
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingData.startMatId[i] = 0;
        tilingData.calMatNum[i] = 0;
    }

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;

    // Distribute batch across cores
    uint32_t baseNum = batchCount / vecCoreNum;
    uint32_t remainNum = batchCount % vecCoreNum;

    uint32_t curMatId = 0;
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        uint32_t curMatNum = 0;
        if (i < remainNum) {
            curMatNum = baseNum + 1;
        } else {
            curMatNum = baseNum;
        }
        tilingData.startMatId[i] = curMatId;
        tilingData.calMatNum[i] = curMatNum;
        curMatId += curMatNum;
    }

    return tilingData;
}


int aclblasCgemvBatched(const std::complex<float> *A, const std::complex<float> *x, std::complex<float> *y,
                        const std::complex<float> &alpha, const int64_t lda,
                        const std::complex<float> &beta, const int64_t incx, const int64_t incy,
                        const int64_t batchCount, const int64_t m, const int64_t n,
                        const int32_t trans,
                        void *stream)
{
    uint32_t numBlocks = 8;
    
    // Data type is always complex<float>
    uint32_t dtype = 1;  // 1 = float
    uint32_t dtypeSize = sizeof(float);

    // Calculate sizes
    size_t matSize = m * n * 2 * dtypeSize;  // Complex matrix
    size_t vecSize = m * 2 * dtypeSize;      // Complex vector (for normal)
    size_t vecSizeTrans = n * 2 * dtypeSize; // Complex vector (for trans)

    size_t inputAByteSize = batchCount * matSize;
    size_t inputXByteSize = batchCount * ((trans == 0) ? vecSize : vecSizeTrans);
    size_t outputYByteSize = batchCount * ((trans == 0) ? vecSize : vecSizeTrans);

    // Mask size for gather operation
    CgemvBatchedTilingData tiling = CalTilingData(batchCount, m, n, dtype, trans, numBlocks);
    uint32_t *mask = CreateCgemvBatchedMask(m, dtype, trans);
    size_t maskSize = tiling.maxMatNum * 32 * 2 * sizeof(uint32_t);  // maxMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM

    size_t workSpaceSize = WORKSPACE_SIZE;

    uint8_t *AHost = reinterpret_cast<uint8_t *>(const_cast<std::complex<float> *>(A));
    uint8_t *xHost = reinterpret_cast<uint8_t *>(const_cast<std::complex<float> *>(x));
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *maskHost = reinterpret_cast<uint8_t *>(mask);
    uint8_t *ADevice = nullptr;
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *maskDevice = nullptr;
    uint8_t *workSpaceDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    // Allocate device memory
    aclrtMalloc((void **)&ADevice, inputAByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&xDevice, inputXByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, outputYByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CgemvBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    // Copy inputs to device
    aclrtMemcpy(ADevice, inputAByteSize, AHost, inputAByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, inputXByteSize, xHost, inputXByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, outputYByteSize, yHost, outputYByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Calculate tiling data
    
    aclrtMemcpy(tilingDevice, sizeof(CgemvBatchedTilingData), &tiling, sizeof(CgemvBatchedTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    cgemv_batched_kernel_do(ADevice, xDevice, maskDevice, yDevice,
                           workSpaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    // Copy output back to host
    aclrtMemcpy(yHost, outputYByteSize, yDevice, outputYByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Apply alpha and beta scaling on host
    // y = alpha * y_temp + beta * y_original
    // Note: For simplicity, we handle the scaling on host side
    // In a production implementation, this should be done on device side
    
    // Create temporary storage for original y
    std::vector<float> yOriginal(outputYByteSize / sizeof(float));
    std::copy(reinterpret_cast<float *>(yHost), 
              reinterpret_cast<float *>(yHost) + outputYByteSize / sizeof(float),
              yOriginal.begin());
    
    // Apply scaling: y = alpha * y_temp + beta * y_original
    float *yFloat = reinterpret_cast<float *>(yHost);
    for (size_t i = 0; i < outputYByteSize / sizeof(float); i += 2) {
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        float yReal = yFloat[i];
        float yImag = yFloat[i + 1];
        float yOrigReal = yOriginal[i];
        float yOrigImag = yOriginal[i + 1];
        
        // alpha * y_temp
        float alphaYReal = alpha.real() * yReal - alpha.imag() * yImag;
        float alphaYImag = alpha.real() * yImag + alpha.imag() * yReal;
        
        // beta * y_original
        float betaYOrigReal = beta.real() * yOrigReal - beta.imag() * yOrigImag;
        float betaYOrigImag = beta.real() * yOrigImag + beta.imag() * yOrigReal;
        
        // y = alpha * y_temp + beta * y_original
        yFloat[i] = alphaYReal + betaYOrigReal;
        yFloat[i + 1] = alphaYImag + betaYOrigImag;
    }

    // Free device memory
    aclrtFree(ADevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(maskDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
