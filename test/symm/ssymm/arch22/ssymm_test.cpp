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

class SsymmArch22Test : public BlasTest<SsymmParam> {};

// null handle: TEST_F (not CSV-driven), per blas-ST-develop convention
TEST_F(SsymmArch22Test, NullHandle)
{
    float alpha = 1.0f;
    float beta  = 1.0f;
    // Minimal dummy host buffers to satisfy non-null pointer check in the
    // direct kernel path; handle=nullptr is caught before any pointer deref.
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ssymm, SsymmArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<SsymmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsymmParam>);

TEST_P(SsymmArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    // ---- Step 1: generate host data ----
    const int64_t aDim   = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    const bool    isUpper = (p.uplo == ACLBLAS_UPPER);

    // A is a symmetric matrix stored in one triangle (row-major, aDim x lda).
    // Use makeBlasArray to fill the full aDim*lda storage; validity of the
    // stored triangle is implicitly defined by uplo in the CPU golden.
    std::vector<float> aHost;
    std::vector<float> bHost;
    std::vector<float> cHost;
    std::vector<float> cResult;

    // Prevent allocation explosion for overflow/rejection test cases.
    // These cases expect ACLBLAS_STATUS_INVALID_VALUE; the API rejects them
    // via parameter validation before pointer dereference, so nullptr is safe.
    constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024; // 256 MB per matrix
    auto safeElemCount = [](int64_t rows, int64_t cols) -> int64_t {
        if (rows <= 0 || cols <= 0) return 0;
        if (cols > kMaxSafeTestElems / rows) return 0;
        return rows * cols;
    };

    const int64_t aElemCount = safeElemCount(aDim, p.lda);
    const int64_t bElemCount = safeElemCount(p.m, p.ldb);
    const int64_t cElemCount = safeElemCount(p.m, p.ldc);

    if (aElemCount > 0) {
        aHost = makeBlasArray(aElemCount, "RANDOM:-10:10", p.randomSeed);
    }
    if (bElemCount > 0) {
        bHost = makeBlasArray(bElemCount, "RANDOM:-10:10", p.randomSeed + 1);
    }
    if (cElemCount > 0) {
        cHost   = makeBlasArray(cElemCount, "RANDOM:-10:10", p.randomSeed + 2);
        cResult = cHost;  // copy to be overwritten by NPU
    }

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr  = p.nullBeta  ? nullptr : &p.beta;
    const float* aPtr     = aHost.empty() ? nullptr : aHost.data();
    const float* bPtr     = bHost.empty() ? nullptr : bHost.data();
    float*       cPtr     = cResult.empty() ? nullptr : cResult.data();

    // ---- Step 2: execute on NPU ----
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch22Test::handle_,
        p.side, p.uplo, p.m, p.n,
        alphaPtr, aPtr, p.lda,
        bPtr,     p.ldb,
        betaPtr,  cPtr, p.ldc);

    // ---- Step 3: verify error code; return early for non-SUCCESS cases ----
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // ---- Step 4: compute golden on CPU ----
    std::vector<float> cGolden = cHost;
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasSsymm_cpu(
        SsymmArch22Test::handle_,
        p.side, p.uplo, p.m, p.n,
        alphaPtr, aPtr, p.lda,
        bPtr,     p.ldb,
        betaPtr,  goldPtr, p.ldc);

    // ---- Step 5: precision comparison (MERE_MARE, community standard) ----
    const size_t cCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc);
    if (cCount == 0) return;

    VerifyConfig cfg;
    cfg.mode           = PrecisionMode::MERE_MARE;
    cfg.mereThreshold  = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}
