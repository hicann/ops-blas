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
static inline int64_t SafeElemCount(int rows, int cols)
{
    if (rows <= 0 || cols <= 0) return 0;
    if (static_cast<int64_t>(cols) > kMaxSafeTestElems / std::max(static_cast<int64_t>(rows), int64_t(1))) return 0;
    return static_cast<int64_t>(rows) * cols;
}

static inline std::vector<float> MakeMatrix(int rows, int cols, int ld,
    uint32_t seed)
{
    int64_t count = SafeElemCount(rows, ld);
    if (count <= 0) return {};
    return makeBlasArray(count, "RANDOM:-10:10", seed);
}

static inline std::vector<float> MakeTriMatrix(int dimA, int lda,
    aclblasFillMode_t uplo, uint32_t seed)
{
    int64_t count = SafeElemCount(dimA, lda);
    if (count <= 0) return {};
    std::string pattern = (uplo == ACLBLAS_UPPER) ? "UPPER" : "LOWER";
    return makeBlasArray(count, "RANDOM:-10:10:" + pattern, seed);
}

static inline void ApplySpecialValues(const std::string& desc,
    std::vector<float>& aHost, std::vector<float>& bHost)
{
    if (desc.find("nanA") != std::string::npos || desc.find("nan_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
    if (desc.find("infA") != std::string::npos || desc.find("inf_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::infinity());
    }
    if (desc.find("nanB") != std::string::npos || desc.find("nan_B") != std::string::npos) {
        std::fill(bHost.begin(), bHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
}

class StrmmArch35Test : public BlasTest<StrmmParam> {};

TEST_F(StrmmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasStrmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Strmm, StrmmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrmmParam>);

TEST_P(StrmmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost = MakeTriMatrix(aDim, p.lda, p.uplo, p.randomSeed);
    std::vector<float> bHost = MakeMatrix(p.m, p.n, p.ldb, p.randomSeed + 1);
    ApplySpecialValues(p.description, aHost, bHost);

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* aPtr     = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    const float* bPtr     = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());

    if (p.m <= 0 || p.n <= 0 || p.lda <= 0 || p.ldb <= 0 || p.ldc <= 0) {
        float dummy = 0.0f;
        aclblasStatus_t ret = aclblasStrmm_npu(
            StrmmArch35Test::handle_,
            p.side, p.uplo, p.trans, p.diag, p.m, p.n,
            alphaPtr, aPtr, p.lda, bPtr, p.ldb, &dummy, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    std::vector<float> cHost = std::vector<float>(static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc), 0.0f);
    float* cPtr = p.nullC ? nullptr : cHost.data();

    aclblasStatus_t ret = aclblasStrmm_npu(
        StrmmArch35Test::handle_,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, cPtr, p.ldc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> cGolden = std::vector<float>(static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc), 0.0f);
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasStatus_t goldenRet = aclblasStrmm_cpu(
        StrmmArch35Test::handle_,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, goldPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    const size_t cCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc);
    if (cCount == 0) return;

    VerifyConfig cfg;
    cfg.mode           = PrecisionMode::MERE_MARE;
    constexpr double kDefaultMereThreshold = 1.0 / 8192.0;
    constexpr double kDefaultMareMultiplier = 10.0;
    cfg.mereThreshold  = (p.mereThreshold > 0.0) ? p.mereThreshold : kDefaultMereThreshold;
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : kDefaultMareMultiplier;
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}
