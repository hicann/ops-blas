/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYMM_PARAM_H
#define CSYMM_PARAM_H

#include <string>
#include <algorithm>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// Parameter struct for aclblasCsymm (complex<float> symmetric matrix product).
//
// A is complex symmetric (A == A^T), not Hermitian: its imaginary part is
// symmetric too and its diagonal carries imaginary information, which is the one
// difference from CHEMM.
//
// Against Cher2kParam, following the BLAS CSYMM spec:
//   - the shape is given by m and n, and A is square of side m or n depending on
//     `side`, so there is no k;
//   - both alpha and beta are complex, because C is a general matrix and nothing
//     constrains it to stay symmetric;
//   - `side` replaces `trans`: it says whether the symmetric A multiplies from the
//     left or the right.
struct CsymmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int m = 0;
    int n = 0;
    aclblasComplex alpha = {1.0f, 0.0f};
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_1");
    int lda = 0;
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_1");
    int ldb = 0;
    aclblasComplex beta = {0.0f, 0.0f};
    BlasFillMode cFill = BlasFillMode("VALUE_NORM_0");
    int ldc = 0;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    bool nullAlpha = false;
    bool nullBeta = false;

    explicit CsymmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));

        alpha.real = parseFloat(ReadMap(map, "alpha_re", "1.0"));
        alpha.imag = parseFloat(ReadMap(map, "alpha_im", "0.0"));

        // A is square of side m (side=LEFT) or n (side=RIGHT); B and C are m x n.
        int defaultLda = (side == ACLBLAS_SIDE_LEFT) ? std::max(1, m) : std::max(1, n);
        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_1"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLda)));
        bFill = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_NORM_1"));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, m))));

        beta.real = parseFloat(ReadMap(map, "beta_re", "0.0"));
        beta.imag = parseFloat(ReadMap(map, "beta_im", "0.0"));
        cFill = BlasFillMode(ReadMap(map, "c_fill", "VALUE_NORM_0"));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");
        nullAlpha = (ReadMap(map, "nullAlpha", "0") == "1");
        nullBeta = (ReadMap(map, "nullBeta", "0") == "1");
    }
};

#endif // CSYMM_PARAM_H
