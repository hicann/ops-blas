/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHER2K_PARAM_H
#define CHER2K_PARAM_H

#include <string>
#include <algorithm>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// Parameter struct for aclblasCher2k (complex<float> Hermitian rank-2k update).
//
// Against CsyrkParam, following the BLAS CHER2K spec:
//   - there are two input matrices, so the CSV carries b_fill and ldb as well;
//   - alpha is complex but beta is *real* (a single CSV column), because a complex
//     beta would not preserve the Hermitian property of C;
//   - trans takes N or C (conjugate transpose), not T.
struct Cher2kParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    aclblasComplex alpha = {1.0f, 0.0f};
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_1");
    int lda = 0;
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_1");
    int ldb = 0;
    float beta = 0.0f;
    BlasFillMode cFill = BlasFillMode("VALUE_NORM_0");
    int ldc = 0;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    bool nullAlpha = false;
    bool nullBeta = false;

    explicit Cher2kParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        alpha.real = parseFloat(ReadMap(map, "alpha_re", "1.0"));
        alpha.imag = parseFloat(ReadMap(map, "alpha_im", "0.0"));

        // Default lda/ldb (column-major); A and B share the same logical shape:
        //   trans=N   -> (n x k), ld >= max(1, n)
        //   trans=C   -> (k x n), ld >= max(1, k)
        int defaultLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_1"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLd)));
        bFill = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_NORM_1"));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(defaultLd)));

        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        cFill = BlasFillMode(ReadMap(map, "c_fill", "VALUE_NORM_0"));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");
        nullAlpha = (ReadMap(map, "nullAlpha", "0") == "1");
        nullBeta = (ReadMap(map, "nullBeta", "0") == "1");
    }
};

#endif // CHER2K_PARAM_H
