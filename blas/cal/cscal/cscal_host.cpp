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
* \file cscal_host.cpp
* \brief
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

constexpr uint32_t MAX_LENG_PER_UB_PROC = 6144;
constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr uint32_t PING_PONG_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t COMPLEX_DATA_NUM_PER_BLOCK = 4;

struct CscalTilingData {
    int32_t n;
    float alphaReal;
    float alphaImag;
};

CscalTilingData CalTilingData(int32_t n, float alphaReal, float alphaImag)
{
    CscalTilingData tilingData;
    tilingData.n = n;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;
    return tilingData;
}

void CreateMaskData(uint32_t *maskData)
{
    uint32_t imagBaseAddr = MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(float) +
                            MAX_LENG_PER_UB_PROC * sizeof(float) * ELEMENTS_EACH_COMPLEX64 * PING_PONG_NUM;
    uint32_t realBaseAddr = imagBaseAddr + MAX_LENG_PER_UB_PROC * sizeof(uint32_t);

    int k = 0;
    for (uint32_t i = 0; i < MAX_LENG_PER_UB_PROC; i++) {
        maskData[k++] = realBaseAddr + i * sizeof(float);
        maskData[k++] = imagBaseAddr + i * sizeof(float);
    }
}

int aclblasCscal(aclblasHandle handle, std::complex<float> *x, const std::complex<float> alpha,
                 const int64_t n, const int64_t incx)
{
    uint32_t numBlocks = 40;

    int32_t totalByteSize = n * sizeof(std::complex<float>);
    int32_t deviceId = 0;

    aclrtStream stream = static_cast<aclrtStream>(handle);

    float alphaReal = alpha.real();
    float alphaImag = alpha.imag();

    CscalTilingData tiling = CalTilingData(n, alphaReal, alphaImag);
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *xDevice = nullptr;
    uint8_t *tilingDevice = nullptr;
    uint8_t *maskDevice = nullptr;

    uint32_t *maskHost = new uint32_t[MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64];
    CreateMaskData(maskHost);

    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(CscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t),
                ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(CscalTilingData), &tiling, sizeof(CscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t),
                maskHost, MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    cscal_kernel_do(xDevice, maskDevice, nullptr, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(tilingDevice);
    aclrtFree(maskDevice);
    delete[] maskHost;

    return ACL_SUCCESS;
}