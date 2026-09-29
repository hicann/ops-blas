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
 * \file cher2k_tiling_data.h
 * \brief CHER2K-specific tiling. Phases 0 and 1 reuse the shared structures in
 *        common/helper/complex_blas3_tiling_data.h.
 */

#ifndef CHER2K_TILING_DATA_H
#define CHER2K_TILING_DATA_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2: assemble the Hermitian C from both orientations of M = A*B^H.
//
//   Mr = Ar*Br^T + Ai*Bi^T          Mi = Ai*Br^T - Ar*Bi^T
//   C  = alpha*M + conj(alpha)*M^H + beta*C_old        (alpha complex, beta real)
// expands to
//   Cr = ar*(Mr + Mr^T) - ai*(Mi + Mi^T) + beta*Cr_old
//   Ci = ar*(Mi - Mi^T) + ai*(Mr - Mr^T) + beta*Ci_old
//
// So the stage needs *both orientations* of Mr and Mi. Unlike CHERK there is no
// free transpose: with A and B distinct, the fourth product is not the transpose
// of the third, so Phase 1 computes all four of Mr, Mr^T, Mi, Mi^T.
//
// Phase 2 reads the temps with a column-major stride, which transposes them, so
// the buffer holding Mr^T is the one whose read yields Mr. The host therefore
// passes each pair swapped; the kernel's parameters are named by what the read
// *yields*, not by what the buffer holds. This matters because Mr - Mr^T is
// antisymmetric: getting the pair backwards flips the sign of the imaginary part.
//
// C is Hermitian, so the diagonal imaginary part is forced to zero.
struct Cher2kCombineTilingData {
    uint32_t n;
    uint32_t ldc;           // leading dimension of C, in complex elements
    uint32_t tempLdc;       // row stride of the four temps, in floats
    uint32_t uploMode;      // CBLAS3_UPLO_UPPER / CBLAS3_UPLO_LOWER
    uint32_t skipAlphaTerm; // alpha == 0 or k == 0: skip the alpha term entirely
    uint32_t isBetaZero;    // beta == 0: do not read C_old
    float alphaRe;
    float alphaIm;
    float betaVal; // beta is real for CHER2K
};

#endif // CHER2K_TILING_DATA_H
