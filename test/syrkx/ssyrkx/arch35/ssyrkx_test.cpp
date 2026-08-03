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
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "verify_uplo.h"
#include "ssyrkx_param.h"
#include "ssyrkx_golden.h"
#include "ssyrkx_npu_wrapper.h"

class SsyrkxArch35Test : public BlasTest<SsyrkxParam> {};

TEST_F(SsyrkxArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    float dummy = 0.0f;
    aclblasStatus_t ret = aclblasSsyrkx_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4,
        &alpha, &dummy, 4, &dummy, 4, &beta, &dummy, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ssyrkx, SsyrkxArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SsyrkxParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsyrkxParam>);

TEST_P(SsyrkxArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Determine matrix dimensions based on trans (A and B have same dimensions in syrkx)
    int aRows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    // Generate data using BlasFillMode from CSV
    uint32_t seedA = p.randomSeed;
    uint32_t seedB = p.randomSeed + 1;
    uint32_t seedC = p.randomSeed + 2;

    std::vector<float> aHost = makeBlasMatrix(aRows, aCols, p.lda, p.a, seedA);
    std::vector<float> bHost = makeBlasMatrix(aRows, aCols, p.ldb, p.b, seedB);
    std::vector<float> cHost = makeBlasMatrix(p.n, p.n, p.ldc, p.c, seedC);

    // Handle nullptr flags for error injection cases
    const float* aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    const float* bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());
    float* cPtr = p.nullC ? nullptr : (cHost.empty() ? nullptr : cHost.data());
    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;

    // Save original C for golden's non-uplo triangle restoration
    std::vector<float> originalC;
    if (cPtr != nullptr && p.n > 0) {
        originalC.assign(cHost.begin(), cHost.end());
    }

    size_t cCount = (p.n > 0) ? static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) : 0;

    // Step 1: NPU execution
    aclblasStatus_t ret = aclblasSsyrkx_npu(
        SsyrkxArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    // Step 2: Check return code
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "case=" << p.caseName << " expected=" << static_cast<int>(p.expectResult)
            << " got=" << static_cast<int>(ret);
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS) << "case=" << p.caseName;

    // Step 3: Quick-return case (n==0) — no computation, C unchanged
    if (p.n == 0) {
        return;
    }

    // Step 4: Golden computation
    std::vector<float> goldenC(originalC);

    aclblasStatus_t goldenRet = aclblasSsyrkx_cpu(
        SsyrkxArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr,
        goldenC.data(), p.ldc, originalC.data());

    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden failed for case=" << p.caseName;

    // Step 5: Verify uplo triangle + non-uplo triangle (unchanged from original)
    VerifyUploTriangle(p, cHost.data(), goldenC.data(), cCount);
}
