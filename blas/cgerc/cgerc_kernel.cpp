/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

#include "cgerc_kernel_impl.h"

__global__ __aicore__ void cgerc(GM_ADDR d_x, GM_ADDR d_y,
                                 GM_ADDR d_offset, GM_ADDR d_A,
                                 GM_ADDR work_space, GM_ADDR tiling_gm)
{
    uint32_t vec_idx = AscendC::GetBlockIdx();

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);
    uint32_t m = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));
    float alphaReal = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 8));
    float alphaImag = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 12));

    uint64_t offset = (*(__gm__ uint64_t *)((__gm__ uint8_t *)tiling_buf + 16 + 8 * vec_idx));
    uint64_t calNum = (*(__gm__ uint64_t *)((__gm__ uint8_t *)tiling_buf + 16 + 40 * 8 + 8 * vec_idx));

    AscendC::SetAtomicAdd<float>();

    AscendC::SetMaskNorm();

    AscendC::GlobalTensor<float> d_x_tensor;
    AscendC::GlobalTensor<float> d_y_tensor;
    AscendC::GlobalTensor<float> d_A_tensor;
    AscendC::GlobalTensor<uint32_t> d_offset_tensor;
 
    d_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_x));
    d_y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_y));
    d_A_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_A));
    d_offset_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(d_offset));
    
    if (calNum > 0) {
        process(d_x_tensor, d_y_tensor, d_offset_tensor, d_A_tensor, alphaReal, alphaImag, m, n, offset, calNum);
    }

    PIPE_BARRIER(ALL);
    AscendC::SetAtomicNone();
}

void cgerc_kernel_do(GM_ADDR d_x, GM_ADDR d_y, GM_ADDR d_offset, GM_ADDR d_A,
                     GM_ADDR work_space, GM_ADDR tiling_gm,
                     uint32_t num_blocks, void *stream)
{
    cgerc<<<num_blocks, nullptr, stream>>>(d_x, d_y, d_offset, d_A, work_space, tiling_gm);
}
