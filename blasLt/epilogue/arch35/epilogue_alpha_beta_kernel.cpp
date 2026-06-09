/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file epilogue_alpha_beta_kernel.cpp
 * \brief SIMT epilogue: D = alpha * D_raw + beta * C.
 */

#include <cstdint>

#include "epilogue_alpha_beta_tiling_data.h"
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_simt.h"

namespace {

constexpr uint32_t SIMT_MAX_THREAD_NUM = 2048;

template <typename TDRaw, typename TC, typename TD>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void EpilogueAlphaBetaSimt(
    uint32_t m, uint32_t n, uint32_t ldc, uint32_t ldd, uint32_t lddRaw, float alpha, float beta, bool useC,
    __gm__ const TDRaw* dRawGm, __gm__ const TC* cGm, __gm__ TD* dGm)
{
    constexpr bool RAW_BF16 = AscendC::IsSameType<TDRaw, bfloat16_t>::value;
    constexpr bool C_BF16 = AscendC::IsSameType<TC, bfloat16_t>::value;
    constexpr bool D_BF16 = AscendC::IsSameType<TD, bfloat16_t>::value;

    const uint32_t total = m * n;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        const uint32_t row = idx / n;
        const uint32_t col = idx % n;
        const uint32_t rawIdx = row * lddRaw + col;
        const uint32_t outIdx = row * ldd + col;

        float rawVal = RAW_BF16 ? __bfloat162float(dRawGm[rawIdx]) : static_cast<float>(dRawGm[rawIdx]);
        float cVal = 0.0f;
        if (useC) {
            const uint32_t cIdx = row * ldc + col;
            cVal = C_BF16 ? __bfloat162float(cGm[cIdx]) : static_cast<float>(cGm[cIdx]);
        }
        const float result = alpha * rawVal + beta * cVal;
        if constexpr (D_BF16) {
            dGm[outIdx] = __float2bfloat16_rn_sat(result);
        } else {
            dGm[outIdx] = static_cast<TD>(result);
        }
    }
}

template <typename TDRaw, typename TC, typename TD>
__aicore__ inline void LaunchEpilogueAlphaBetaTyped(
    const EpilogueAlphaBetaTilingData& tiling, GM_ADDR dRaw, GM_ADDR c, GM_ADDR d)
{
    auto* dRawGm = reinterpret_cast<__gm__ const TDRaw*>(dRaw);
    auto* cGm = reinterpret_cast<__gm__ const TC*>(c);
    auto* dGm = reinterpret_cast<__gm__ TD*>(d);
    const bool useC = tiling.useC != 0U;
    asc_vf_call<EpilogueAlphaBetaSimt<TDRaw, TC, TD>>(
        dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.ldc, tiling.ldd, tiling.lddRaw, tiling.alpha,
        tiling.beta, useC, dRawGm, cGm, dGm);
}

__aicore__ inline void DispatchEpilogueAlphaBeta(
    const EpilogueAlphaBetaTilingData& tiling, GM_ADDR dRaw, GM_ADDR c, GM_ADDR d)
{
    const bool rawBf16 = tiling.dtypeDRaw != 0U;
    const bool cBf16 = tiling.dtypeC != 0U;
    const bool dBf16 = tiling.dtypeD != 0U;

    if (!rawBf16 && !cBf16 && !dBf16) {
        LaunchEpilogueAlphaBetaTyped<float, float, float>(tiling, dRaw, c, d);
    } else if (!rawBf16 && !cBf16 && dBf16) {
        LaunchEpilogueAlphaBetaTyped<float, float, bfloat16_t>(tiling, dRaw, c, d);
    } else if (!rawBf16 && cBf16 && !dBf16) {
        LaunchEpilogueAlphaBetaTyped<float, bfloat16_t, float>(tiling, dRaw, c, d);
    } else if (!rawBf16 && cBf16 && dBf16) {
        LaunchEpilogueAlphaBetaTyped<float, bfloat16_t, bfloat16_t>(tiling, dRaw, c, d);
    } else if (rawBf16 && !cBf16 && !dBf16) {
        LaunchEpilogueAlphaBetaTyped<bfloat16_t, float, float>(tiling, dRaw, c, d);
    } else if (rawBf16 && !cBf16 && dBf16) {
        LaunchEpilogueAlphaBetaTyped<bfloat16_t, float, bfloat16_t>(tiling, dRaw, c, d);
    } else if (rawBf16 && cBf16 && !dBf16) {
        LaunchEpilogueAlphaBetaTyped<bfloat16_t, bfloat16_t, float>(tiling, dRaw, c, d);
    } else {
        LaunchEpilogueAlphaBetaTyped<bfloat16_t, bfloat16_t, bfloat16_t>(tiling, dRaw, c, d);
    }
}

} // namespace

__global__ __aicore__ void EpilogueAlphaBetaKernel(GM_ADDR dRaw, GM_ADDR c, GM_ADDR d, EpilogueAlphaBetaTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DispatchEpilogueAlphaBeta(tiling, dRaw, c, d);
}

void epilogue_alpha_beta_kernel_do(
    uint8_t* dRaw, uint8_t* c, uint8_t* d, const EpilogueAlphaBetaTilingData& tiling, void* stream)
{
    EpilogueAlphaBetaTilingData tilingCopy = tiling;
    EpilogueAlphaBetaKernel<<<tilingCopy.usedCoreNum, nullptr, stream>>>(dRaw, c, d, tilingCopy);
}
