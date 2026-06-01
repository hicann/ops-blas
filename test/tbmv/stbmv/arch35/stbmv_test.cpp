/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "stbmv_param.h"
#include "stbmv_golden.h"
#include "stbmv_npu_wrapper.h"

inline std::vector<float> MakeBandMatrix(int n, int k, int lda, bool upper, bool unitDiag, uint32_t seed)
{
    const size_t aSize = (n > 0) ? static_cast<size_t>(lda) * n : 1;
    std::vector<float> a(aSize, 0.0f);
    if (n <= 0) return a;

    std::mt19937 rng(seed ? seed : 42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int col = 0; col < n; ++col) {
        int rowStart = upper ? std::max(0, col - k) : col;
        int rowEnd = upper ? col : std::min(n - 1, col + k);
        for (int row = rowStart; row <= rowEnd; ++row) {
            size_t idx = upper
                ? static_cast<size_t>(k + row - col) + static_cast<size_t>(col) * lda
                : static_cast<size_t>(row - col) + static_cast<size_t>(col) * lda;
            a[idx] = (row == col && unitDiag) ? 1.0f : dist(rng);
        }
    }
    return a;
}

inline std::vector<float> MakeStrided(int count, int inc, uint32_t seed)
{
    if (count <= 0) return {};
    int absInc = std::abs(inc);
    size_t size = static_cast<size_t>((count - 1) * absInc + 1);
    std::vector<float> data(size, 0.0f);

    std::mt19937 rng(seed ? seed : 42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int i = 0; i < count; ++i) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = dist(rng);
    }
    return data;
}

class StbmvArch35Test : public BlasTest<StbmvParam> { };

INSTANTIATE_TEST_SUITE_P(
    Stbmv, StbmvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StbmvParam>);

TEST_P(StbmvArch35Test, CsvDriven) {
    const auto& p = GetParam();

    bool isUpper = (p.uplo == ACLBLAS_UPPER);
    bool isUnit = (p.diag == ACLBLAS_UNIT);
    auto a = MakeBandMatrix(p.n, p.k, p.lda, isUpper, isUnit, p.randomSeed);
    auto x = MakeStrided(p.n, p.incx, p.randomSeed + 1);
    std::vector<float> golden = x;

    aclblasStatus_t ret = aclblasStbmv_npu(
        StbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        a.data(), p.lda, x.data(), p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStbmv_cpu(
        StbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        a.data(), p.lda, golden.data(), p.incx);

    const int absIncx = std::abs(p.incx);
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    const float* outPtr  = (p.incx < 0 && p.n > 0) ? x.data() + (p.n - 1) * absIncx : x.data();
    const float* goldPtr = (p.incx < 0 && p.n > 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();

    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr,
        static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
}
