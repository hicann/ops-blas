/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "common/helper/gemm_strided_tiling_arch35.h"

enum class LtRegularMatmulType : uint8_t {
    NONE,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP8_E4M3_E5M2,
    FP8_E5M2_E4M3,
    INT8,
};

struct LtInt32EpilogueTiling {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t ldd;
    uint32_t blocks;
    uint32_t threads;
    int32_t alpha;
    int32_t beta;
};

void matmul_regular_cube_do(
    uint8_t* a, uint8_t* b, uint8_t* output, const GemmStridedBatchedExTilingData& tiling,
    LtRegularMatmulType type, bool directOutput, void* stream);
void matmul_int32_epilogue_do(
    uint8_t* raw, uint8_t* c, uint8_t* d, const LtInt32EpilogueTiling& tiling, void* stream);
