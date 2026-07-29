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
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssyr2k_param.h"
#include "ssyr2k_golden.h"
#include "ssyr2k_npu_wrapper.h"

class Ssyr2kArch35Test : public BlasTest<Ssyr2kParam> {};

static inline void VerifySsyr2kUploTriangle(
    const Ssyr2kParam& p, const float* cPtr, const float* goldPtr, size_t cSize)
{
    if (cSize == 0) return;

    std::vector<float> npuUplo;
    std::vector<float> goldenUplo;
    std::vector<float> npuNonUplo;
    std::vector<float> goldenNonUplo;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUplo.push_back(cPtr[idx]);
                goldenUplo.push_back(goldPtr[idx]);
            } else {
                npuNonUplo.push_back(cPtr[idx]);
                goldenNonUplo.push_back(goldPtr[idx]);
            }
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenUplo.data(), goldenUplo.size());
    EXPECT_TRUE(Verifier::verifyVector(npuUplo.data(), goldenUplo.data(), npuUplo.size(), 1, cfg, p.caseName));

    if (!npuNonUplo.empty()) {
        VerifyConfig cfgNonUplo;
        applyMixedTolerance(cfgNonUplo, ACL_FLOAT, goldenNonUplo.data(), goldenNonUplo.size());
        EXPECT_TRUE(Verifier::verifyVector(npuNonUplo.data(), goldenNonUplo.data(),
            npuNonUplo.size(), 1, cfgNonUplo, std::string(p.caseName) + "_nonuplo"));
    }
}

TEST_F(Ssyr2kArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsyr2k_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ssyr2k, Ssyr2kArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<Ssyr2kParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Ssyr2kParam>);

TEST_P(Ssyr2kArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Compute A/B matrix dimensions based on trans
    int aRows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    // Generate input data (column-major: need lda * numCols elements)
    const size_t aSize = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols);
    const size_t bSize = static_cast<size_t>(p.ldb) * static_cast<size_t>(aCols);
    const size_t cSize = static_cast<size_t>(p.n) * static_cast<size_t>(p.ldc);

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(aSize), p.fillA, p.randomSeed);
    std::vector<float> bHost = makeBlasArray(static_cast<int64_t>(bSize), p.fillB, p.randomSeed + 1);
    std::vector<float> cHost = makeBlasArray(static_cast<int64_t>(cSize), p.fillC, p.randomSeed + 2);

    // Save original C data for golden computation BEFORE NPU call modifies it
    std::vector<float> cGolden(cHost);

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr  = p.nullBeta  ? nullptr : &p.beta;
    const float* aPtr     = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    const float* bPtr     = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());
    float*       cPtr     = p.nullC ? nullptr : (cHost.empty() ? nullptr : cHost.data());

    // For n <= 0 or k < 0 or other invalid param cases, pass through to operator
    // (npu_wrapper handles these by direct pass-through)
    if (p.n <= 0 || p.k < 0) {
        aclblasStatus_t ret = aclblasSsyr2k_npu(
            Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
            alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasSsyr2k_npu(
        Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // Compute golden (using original C data saved before NPU call)
    // cGolden was already copied from cHost BEFORE the NPU call modified cHost
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasStatus_t goldenRet = aclblasSsyr2k_cpu(
        Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, goldPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    const size_t cCount = cSize;
    VerifySsyr2kUploTriangle(p, cPtr, goldPtr, cCount);
}
