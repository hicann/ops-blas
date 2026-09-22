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
 * \file cherk_kernel.h
 * \brief Kernel launcher signatures for CHERK on arch22, shared by the host and
 *        the kernel translation unit so that the compiler checks both against one
 *        declaration.
 *
 *        GM_ADDR is guarded: on the host side it falls back to a plain pointer,
 *        while the kernel translation unit includes kernel_operator.h first and
 *        therefore keeps AscendC's __gm__ qualified definition. This is the same
 *        convention other arch22 operators in this repository follow.
 */

#ifndef CHERK_ARCH22_KERNEL_H
#define CHERK_ARCH22_KERNEL_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"
#include "cherk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): complex A -> packed real Ar, Ai.
void cherk_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling);

// Phase 0 for trans='N': two K-interleaved operands P = [Ai_j, Ar_j] and
// Q = [Ar_j, -Ai_j], whose Gram products yield the real and imaginary parts with
// the imaginary part's signs alternating inside the accumulator.
void cherk_split_interleave_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR p, GM_ADDR q, CBlas3SplitConcatTilingData tiling);

// Phase 1 (AIC): one real GEMM. tiling.transMode picks the __global__ entry whose
// MatmulType has isTrans baked into the operand that needs transposing.
void cherk_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling);

// Phase 2 (AIV): assemble Hermitian C from t1..t4 and apply alpha/beta.
//
// Phase 2 reads the temps with a column-major stride, which transposes them, so
// the caller must pass the buffer holding t4 in the t3 slot and vice versa (see
// the layout invariants at the top of cherk_kernel.cpp).
void cherk_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c,
    CherkCombineTilingData tiling);

// Compile-time singleCoreK of the Phase 1 static tiling. A launch whose runtime K
// exceeds it is silently mis-computed, so the host splits K to this value.
uint32_t cherk_gemm_single_k();

#endif // CHERK_ARCH22_KERNEL_H
