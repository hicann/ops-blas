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

#include <algorithm>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct Cher2kParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    float beta = 0.0f;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    int lda = 1;
    int ldb = 1;
    int ldc = 1;
    BlasFillMode fillA = BlasFillMode("RANDOM_NORM_5_5");
    BlasFillMode fillB = BlasFillMode("RANDOM_NORM_5_5");
    BlasFillMode fillC = BlasFillMode("RANDOM_NORM_5_5");

    static int ParseLd(const csv_map& map, const char* key, int defaultValue)
    {
        const std::string value = ReadMap(map, key, "");
        return value.empty() ? defaultValue : parseInt(value);
    }

    explicit Cher2kParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        const std::string alphaRealString = ReadMap(map, "alpha_real", "1");
        nullAlpha = alphaRealString == "null" || alphaRealString == "nullptr";
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealString, 1.0f);
        alphaImag = parseFloat(ReadMap(map, "alpha_imag", "0"), 0.0f);

        const std::string betaString = ReadMap(map, "beta", "0");
        nullBeta = betaString == "null" || betaString == "nullptr";
        beta = nullBeta ? 0.0f : parseFloat(betaString, 0.0f);

        fillA = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        fillB = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));
        fillC = BlasFillMode(ReadMap(map, "c_fill", "RANDOM_NORM_5_5"));
        const int defaultLd = trans == ACLBLAS_OP_N ? std::max(1, n) : std::max(1, k);
        lda = ParseLd(map, "lda", defaultLd);
        ldb = ParseLd(map, "ldb", defaultLd);
        ldc = ParseLd(map, "ldc", std::max(1, n));

        nullA = ReadMap(map, "nullA", "0") == "1";
        nullB = ReadMap(map, "nullB", "0") == "1";
        nullC = ReadMap(map, "nullC", "0") == "1";
    }
};
