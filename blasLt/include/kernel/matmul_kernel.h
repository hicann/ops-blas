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
 * \file matmul_kernel.h
 * \brief Host-side launch declarations for blasLt matmul device kernels.
 *
 * Implementations are provided under type-specific directories (for example `blasLt/matmul_fp32/`); this header
 * collects their entry points as additional precisions (FP16, MXFP8, and others) are integrated.
 */

#pragma once

#include <cstdint>

#include "matmul_tiling_data.h"

// FP32 x FP32 -> FP32
void matmul_fp32_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* dRaw, const MatmulFp32TilingData& tiling, uint32_t numBlocks, void* stream);

// MXFP8 (BLAZE)
void matmul_mxfp8_kernel_do(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    bool transA, bool transB, void* stream);

// MXFP4 (BLAZE)
void matmul_mxfp4_kernel_do_e2m1_e2m1_fp32(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    bool transA, bool transB, void* stream);
void matmul_mxfp4_kernel_do_e2m1_e2m1_bf16(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    bool transA, bool transB, void* stream);
