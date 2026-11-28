/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define ASCENDC_CUBE_ONLY
#include "matmul_regular_kernel.h"
#include "common/helper/gemm_strided_cube_arch35.h"

#define LT_REGULAR_FLOAT_KERNELS(NAME, A, B, D, BM, BK, BN, C0, QUANT)             \
    GEMM_STRIDED_CUBE_KERNEL(NAME##_direct, A, B, D, float, BM, BK, BN, C0, QUANT) \
    GEMM_STRIDED_CUBE_KERNEL(NAME##_raw, A, B, float, float, BM, BK, BN, C0, QuantMode_t::NoQuant)

LT_REGULAR_FLOAT_KERNELS(lt_regular_fp16, half, half, half, 128, 16, 128, 16, QuantMode_t::F322F16)
LT_REGULAR_FLOAT_KERNELS(lt_regular_bf16, bfloat16_t, bfloat16_t, bfloat16_t, 128, 16, 128, 16, QuantMode_t::F322BF16)
LT_REGULAR_FLOAT_KERNELS(lt_regular_e4m3, fp8_e4m3fn_t, fp8_e4m3fn_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
LT_REGULAR_FLOAT_KERNELS(lt_regular_e5m2, fp8_e5m2_t, fp8_e5m2_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
LT_REGULAR_FLOAT_KERNELS(lt_regular_e4m3_e5m2, fp8_e4m3fn_t, fp8_e5m2_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
LT_REGULAR_FLOAT_KERNELS(lt_regular_e5m2_e4m3, fp8_e5m2_t, fp8_e4m3fn_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
#undef LT_REGULAR_FLOAT_KERNELS

GEMM_STRIDED_CUBE_KERNEL(
    lt_regular_int8, int8_t, int8_t, int32_t, int32_t, 128, 32, 128, 32, QuantMode_t::NoQuant)

#define LT_LAUNCH_FLOAT_KERNEL(NAME)                                             \
    if (directOutput) {                                                          \
        NAME##_direct<<<blocks, nullptr, stream>>>(a, b, output, tiling);         \
    } else {                                                                     \
        NAME##_raw<<<blocks, nullptr, stream>>>(a, b, output, tiling);            \
    }

void matmul_regular_cube_do(
    uint8_t* a, uint8_t* b, uint8_t* output, const GemmStridedBatchedExTilingData& tiling,
    LtRegularMatmulType type, bool directOutput, void* stream)
{
    const uint32_t blocks = static_cast<uint32_t>(tiling.usedCoreNum);
    switch (type) {
        case LtRegularMatmulType::FP16:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_fp16)
            break;
        case LtRegularMatmulType::BF16:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_bf16)
            break;
        case LtRegularMatmulType::FP8_E4M3:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_e4m3)
            break;
        case LtRegularMatmulType::FP8_E5M2:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_e5m2)
            break;
        case LtRegularMatmulType::FP8_E4M3_E5M2:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_e4m3_e5m2)
            break;
        case LtRegularMatmulType::FP8_E5M2_E4M3:
            LT_LAUNCH_FLOAT_KERNEL(lt_regular_e5m2_e4m3)
            break;
        case LtRegularMatmulType::INT8:
            lt_regular_int8<<<blocks, nullptr, stream>>>(a, b, output, tiling);
            break;
        default:
            break;
    }
}
#undef LT_LAUNCH_FLOAT_KERNEL
