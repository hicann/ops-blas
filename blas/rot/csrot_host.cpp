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
 * \file csrot_host.cpp
 * \brief Complex vector rotation host implementation
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../utils/aclblas_kernel_do.h"

using aclblasHandle = void *;

// Tiling data structure from sip
struct RotTilingData {
    int32_t elementCount;
    float cosValue;
    float sinValue;
};


int aclblasCsrot(float *x, float *y, const int64_t n, const float c, const float s, void *stream)
{
    uint32_t numBlocks = 8;

    size_t vectorByteSize = n * sizeof(float);
    size_t workSpaceSize = 32;  // 32 bytes workspace

    int32_t deviceId = 0;

    // Prepare tiling data
    RotTilingData tiling;
    tiling.elementCount = static_cast<int32_t>(n);
    tiling.cosValue = c;
    tiling.sinValue = s;

    uint8_t *xHost = reinterpret_cast<uint8_t *>(x);
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *workSpaceDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void **)&xDevice, vectorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, vectorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tilingDevice, sizeof(RotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDevice, vectorByteSize, xHost, vectorByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, vectorByteSize, yHost, vectorByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(RotTilingData), &tiling, sizeof(RotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    csrot_kernel_do(xDevice, yDevice, workSpaceDevice, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(xHost, vectorByteSize, xDevice, vectorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(yHost, vectorByteSize, yDevice, vectorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
