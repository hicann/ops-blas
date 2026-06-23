/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssymm_param.h"
#include "ssymm_golden.h"
#include "ssymm_npu_wrapper.h"

class SsymmArch35Test : public BlasTest<SsymmParam> {};

TEST_F(SsymmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta  = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ssymm, SsymmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SsymmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsymmParam>);

TEST_P(SsymmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost = makeBlasMatrixRM(aDim, p.n, p.lda, "RANDOM:-10:10", p.randomSeed);
    std::vector<float> bHost = makeBlasMatrixRM(p.m, p.n, p.ldb, "RANDOM:-10:10", p.randomSeed + 1);
    std::vector<float> cHost = makeBlasMatrixRM(p.m, p.n, p.ldc, "RANDOM:-10:10", p.randomSeed + 2);
    std::vector<float> cResult = cHost;

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr  = p.nullBeta  ? nullptr : &p.beta;
    const float* aPtr     = aHost.empty() ? nullptr : aHost.data();
    const float* bPtr     = bHost.empty() ? nullptr : bHost.data();
    float*       cPtr     = cResult.empty() ? nullptr : cResult.data();

    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_,
        p.side, p.uplo, p.m, p.n,
        alphaPtr, aPtr, p.lda,
        bPtr,     p.ldb,
        betaPtr,  cPtr, p.ldc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> cGolden = cHost;
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasSsymm_cpu(
        SsymmArch35Test::handle_,
        p.side, p.uplo, p.m, p.n,
        alphaPtr, aPtr, p.lda,
        bPtr,     p.ldb,
        betaPtr,  goldPtr, p.ldc);

    if (p.m == 0 || p.n == 0) return;
    const size_t cCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc);

    VerifyConfig cfg;
    cfg.mode           = PrecisionMode::MERE_MARE;
    cfg.mereThreshold  = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}
