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
 * \file cher2k_kernel.h
 * \brief Kernel launcher signatures for CHER2K on arch22, shared by the host and
 *        the kernel translation unit so that the compiler checks both against one
 *        declaration.
 */

#ifndef CHER2K_ARCH22_KERNEL_H
#define CHER2K_ARCH22_KERNEL_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"
#include "cher2k_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): complex input -> packed real parts. Called twice, once for A and
// once for B. Shared implementation.
void cher2k_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling);

// Between the two groups of Phase 1 GEMMs: flip the sign of the packed Bi buffer
// in place. See NegateBody in the shared header for why.
void cher2k_negate_kernel_do(uint32_t blockDim, void* stream, GM_ADDR buf, uint32_t count);

// Phase 1 (AIC): one real GEMM. Dispatches on tiling.transMode.
void cher2k_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling);

// Phase 2 (AIV): Hermitian assembly and scaling.
//
// The four temp parameters are named by what a column-major read of them
// *yields*, not by what Phase 1 wrote into them. Because that read transposes,
// the caller passes the buffer holding Mr^T as `mrSrc` and the one holding Mr as
// `mrtSrc` (and likewise for the imaginary pair). Getting the pair backwards
// flips the sign of the imaginary part, which is antisymmetric.
void cher2k_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR mrSrc, GM_ADDR mrtSrc, GM_ADDR miSrc, GM_ADDR mitSrc, GM_ADDR c,
    Cher2kCombineTilingData tiling);

// The compile-time singleK of the Matmul static tiling, so the host can split K
// without duplicating the constant.
uint32_t cher2k_gemm_single_k();

#endif // CHER2K_ARCH22_KERNEL_H
