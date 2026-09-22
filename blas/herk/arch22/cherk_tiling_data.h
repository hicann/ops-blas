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
 * \file cherk_tiling_data.h
 * \brief CHERK-specific tiling. Phases 0 and 1 reuse the shared structures in
 *        common/helper/complex_blas3_tiling_data.h.
 */

#ifndef CHERK_TILING_DATA_H
#define CHERK_TILING_DATA_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2: assemble the Hermitian C from the four real temps.
//
//   t1 = Ar*Ar^T, t2 = Ai*Ai^T, t3 = Ai*Ar^T, t4 = Ar*Ai^T
//   Cr = alpha*(t1 + t2) + beta*Cr_old
//   Ci = alpha*(t3 - t4) + beta*Ci_old        (alpha and beta both real)
//
// Phase 2 reads the temps with a column-major stride, which transposes them. That
// is harmless for the symmetric t1 and t2, and for the imaginary pair it is
// exploited: t4 = t3^T holds exactly, so the host swaps the t3 / t4 pointers and
// the transposed read yields the orientation the formula needs at zero cost.
//
// C is Hermitian, so the diagonal imaginary part is forced to zero.
struct CherkCombineTilingData {
    uint32_t n;
    uint32_t ldc;           // leading dimension of C, in complex elements
    uint32_t tempLdc;       // row stride of the four temps, in floats
    uint32_t uploMode;      // CBLAS3_UPLO_UPPER / CBLAS3_UPLO_LOWER
    uint32_t skipAlphaTerm; // alpha == 0 or k == 0: skip the alpha term entirely
    uint32_t isBetaZero;    // beta == 0: do not read C_old
    // 1 when Phase 1 produced the real and imaginary parts complete (K-interleaved
    // path): t1 holds Pr, t3 holds Pi, and t2 / t4 are unused.
    uint32_t fusedTemps;
    float alphaVal;
    float betaVal;
};

#endif // CHERK_TILING_DATA_H
