/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHEMM_TILING_DATA_H
#define CHEMM_TILING_DATA_H

#include <cstdint>
#include "cann_ops_blas_common.h"

struct ChemmMmadTiling {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t sideMode;
    uint32_t uploMode;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
};

// Workspace: A_r(M*K) + A_i(M*K) + B_r(K*N) + B_i(K*N) + RR(M*N) + II(M*N) + RI(M*N) + IR(M*N)
static inline size_t CalcChemmWorkspaceBytes(uint32_t m, uint32_t n, uint32_t k)
{
    uint64_t mk = static_cast<uint64_t>(m) * k;
    uint64_t kn = static_cast<uint64_t>(k) * n;
    uint64_t mn = static_cast<uint64_t>(m) * n;
    uint64_t total = 2ULL * mk + 2ULL * kn + 4ULL * mn;
    size_t bytes = static_cast<size_t>(total) * sizeof(float);
    return (bytes + 31U) & ~static_cast<size_t>(31);
}

#endif // CHEMM_TILING_DATA_H
