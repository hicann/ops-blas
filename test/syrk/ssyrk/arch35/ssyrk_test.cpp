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
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "ssyrk_param.h"
#include "ssyrk_golden.h"
#include "ssyrk_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class SsyrkArch35Test : public BlasTest<SsyrkParam> {
};

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(SsyrkArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasSsyrk_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4,
        &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}

INSTANTIATE_TEST_SUITE_P(
    Ssyrk, SsyrkArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SsyrkParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsyrkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct SsyrkHostData {
    std::vector<float> aHost;
    std::vector<float> cHost;
    std::vector<float> cGolden;
    const float* aPtr = nullptr;
    float* cPtr = nullptr;
    float* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const SsyrkParam& p, SsyrkHostData& d)
{
    const int aRows = p.lda;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(float);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(float);
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    if (3 * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (3 * cBytes + aBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        d.aHost = makeBlasMatrix(aRows, aCols, p.lda, p.aFill, p.randomSeed);
        d.cHost = makeBlasMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 1);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);

    if (d.cPtr != nullptr) {
        try {
            d.cGolden = d.cHost;
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
    }
    return true;
}

static void VerifySsyrkResult(const SsyrkParam& p, const SsyrkHostData& d)
{
    if (d.cCount == 0) return;

    std::vector<float> npuUplo;
    std::vector<float> goldenUplo;
    std::vector<float> npuNonUplo;
    std::vector<float> goldenNonUplo;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUplo.push_back(d.cPtr[idx]);
                goldenUplo.push_back(d.cGolden[idx]);
            } else {
                npuNonUplo.push_back(d.cPtr[idx]);
                goldenNonUplo.push_back(d.cGolden[idx]);
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

TEST_P(SsyrkArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    SsyrkHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    aclblasStatus_t ret = aclblasSsyrk_npu(
        SsyrkArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    aclblasStatus_t goldenRet = aclblasSsyrk_cpu(
        SsyrkArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    VerifySsyrkResult(p, d);
}
