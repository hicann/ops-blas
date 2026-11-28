/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_regular_kernel.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace {
constexpr uint32_t MAX_THREADS = 2048;

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS) inline void MatmulInt32EpilogueVf(
    __gm__ const uint32_t* raw, __gm__ const uint32_t* c, __gm__ uint32_t* d, LtInt32EpilogueTiling tiling)
{
    const uint64_t total = static_cast<uint64_t>(tiling.m) * tiling.n;
    const uint64_t step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += step) {
        const uint64_t row = idx / tiling.n;
        const uint64_t col = idx % tiling.n;
        // Unsigned arithmetic preserves the exact INT32 result when it is representable.
        uint32_t result = static_cast<uint32_t>(tiling.alpha) * raw[idx];
        if (tiling.beta != 0) {
            result += static_cast<uint32_t>(tiling.beta) * c[row * tiling.ldc + col];
        }
        d[row * tiling.ldd + col] = result;
    }
}
} // namespace

__global__ __aicore__ void MatmulInt32EpilogueKernel(GM_ADDR raw, GM_ADDR c, GM_ADDR d, LtInt32EpilogueTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<MatmulInt32EpilogueVf>(
        dim3{
            tiling.threads, 1, 1
        },
        reinterpret_cast<__gm__ const uint32_t*>(raw),
        reinterpret_cast<__gm__ const uint32_t*>(c), reinterpret_cast<__gm__ uint32_t*>(d), tiling);
}

void matmul_int32_epilogue_do(
    uint8_t* raw, uint8_t* c, uint8_t* d, const LtInt32EpilogueTiling& tiling, void* stream)
{
    LtInt32EpilogueTiling args = tiling;
    MatmulInt32EpilogueKernel<<<args.blocks, nullptr, stream>>>(raw, c, d, args);
}
