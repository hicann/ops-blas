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

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

#include "cann_ops_blas.h"
#include "types.h"

static aclblasOperation_t parseTrans(const std::string& trans) {
    if (trans == "N") return ACLBLAS_OP_N;
    if (trans == "T") return ACLBLAS_OP_T;
    if (trans == "C") return ACLBLAS_OP_C;
    return ACLBLAS_OP_N;
}

static void gbmv_golden_impl(const TestCaseConfig& tc,
                               const std::vector<std::vector<float>>& inputs,
                               std::vector<float>& goldenOutput) {
    aclblasOperation_t trans = parseTrans(tc.trans.value_or("N"));
    int64_t m = tc.m.value_or(8);
    int64_t n = tc.n.value_or(8);
    int64_t kl = tc.kl.value_or(2);
    int64_t ku = tc.ku.value_or(2);
    int64_t lda = tc.lda.value_or(kl + ku + 1);
    int64_t incx = tc.incx.value_or(1);
    int64_t incy = tc.incy.value_or(1);
    float alpha = tc.alphaReal.value_or(1.0f);
    float beta = tc.betaReal.value_or(0.0f);

    const float* A = inputs[0].data();

    bool isTransN = (trans == ACLBLAS_OP_N);
    int64_t xCount = isTransN ? n : m;
    int64_t yCount = isTransN ? m : n;
    int64_t absIncx = (incx < 0) ? -incx : incx;
    int64_t absIncy = (incy < 0) ? -incy : incy;

    const float* xBase = inputs[1].data();
    const float* x = (incx < 0 && xCount > 0)
        ? (xBase + (xCount - 1) * absIncx) : xBase;

    const float* yInitBase = inputs[3].data();
    const float* yInit = (incy < 0 && yCount > 0)
        ? (yInitBase + (yCount - 1) * absIncy) : yInitBase;

    size_t ySize = (yCount > 0) ? static_cast<size_t>((yCount - 1) * absIncy + 1) : 1;
    goldenOutput.resize(ySize, 0.0f);

    float* yGolden = goldenOutput.data();

    if (yCount == 0) return;

    if (trans == ACLBLAS_OP_N) {
        for (int64_t i = 0; i < m; i++) {
            double sum = 0.0;
            int64_t jStart = (i > kl) ? (i - kl) : 0;
            int64_t jEnd = (i + ku < n - 1) ? (i + ku) : (n - 1);
            for (int64_t j = jStart; j <= jEnd; j++) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[j * incx]);
            }
            yGolden[i * absIncy] = static_cast<float>(
                static_cast<double>(alpha) * sum +
                static_cast<double>(beta) * static_cast<double>(yInit[i * incy]));
        }
    } else {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            int64_t firstRow = (j > ku) ? (j - ku) : 0;
            int64_t lastRow = (j + kl < m - 1) ? (j + kl) : (m - 1);
            for (int64_t i = firstRow; i <= lastRow; i++) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[i * incx]);
            }
            yGolden[j * absIncy] = static_cast<float>(
                static_cast<double>(alpha) * sum +
                static_cast<double>(beta) * static_cast<double>(yInit[j * incy]));
        }
    }
}