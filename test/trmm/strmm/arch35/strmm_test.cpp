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
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strmm_param.h"
#include "strmm_golden.h"
#include "strmm_npu_wrapper.h"

constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024;
static inline int64_t SafeElemCount(int64_t rows, int64_t cols)
{
    if (rows <= 0 || cols <= 0) return 0;
    if (cols > kMaxSafeTestElems / std::max(rows, int64_t(1))) return 0;
    return rows * cols;
}

static inline std::vector<float> MakeMatrix(int64_t rows, int64_t cols, int64_t ld,
    uint32_t seed)
{
    int64_t count = SafeElemCount(rows, ld);
    if (count <= 0) return {};
    return makeBlasArray(count, "RANDOM:-10:10", seed);
}

static inline std::vector<float> MakeTriMatrix(int64_t dimA, int64_t lda,
    aclblasFillMode_t uplo, uint32_t seed)
{
    int64_t count = SafeElemCount(dimA, lda);
    if (count <= 0) return {};
    std::string pattern = (uplo == ACLBLAS_UPPER) ? "UPPER" : "LOWER";
    return makeBlasArray(count, "RANDOM:-10:10:" + pattern, seed);
}

class StrmmArch35Test : public BlasTest<StrmmParam> {};

TEST_F(StrmmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    aclblasStatus_t ret = aclblasStrmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4,
        &alpha, a.data(), 4, b.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Strmm, StrmmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrmmParam>);

TEST_P(StrmmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost = MakeTriMatrix(aDim, p.lda, p.uplo, p.randomSeed);
    std::vector<float> bHost = MakeMatrix(p.m, p.n, p.ldb, p.randomSeed + 1);
    if (p.caseName.find("nan") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
    std::vector<float> bResult = bHost;

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* aPtr     = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    float*       bPtr     = p.nullB ? nullptr : (bResult.empty() ? nullptr : bResult.data());

    aclblasStatus_t ret = aclblasStrmm_npu(
        StrmmArch35Test::handle_,
        p.side, p.uplo, p.transA, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> bGolden = bHost;
    float* goldPtr = bGolden.empty() ? nullptr : bGolden.data();

    aclblasStrmm_cpu(
        StrmmArch35Test::handle_,
        p.side, p.uplo, p.transA, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, goldPtr, p.ldb);

    const size_t bCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldb);
    if (bCount == 0) return;

    VerifyConfig cfg;
    cfg.mode           = PrecisionMode::MERE_MARE;
    cfg.mereThreshold  = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
    EXPECT_TRUE(Verifier::verifyVector(bPtr, goldPtr, bCount, 1, cfg, p.caseName));
}
