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
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "aclblasSgeqrfBatched_param.h"
#include "aclblasSgeqrfBatched_golden.h"
#include "aclblasSgeqrfBatched_npu_wrapper.h"

class AclblasSgeqrfBatchedArch35Test : public BlasTest<AclblasSgeqrfBatchedParam> {};

// Null handle test — separate TEST_F per convention
TEST_F(AclblasSgeqrfBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSgeqrfBatched_npu(nullptr, 8, 8, nullptr, 8, nullptr, nullptr, 2);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    AclblasSgeqrfBatched, AclblasSgeqrfBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<AclblasSgeqrfBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AclblasSgeqrfBatchedParam>);

static std::vector<float> GenerateTestMatrix(const AclblasSgeqrfBatchedParam& p, int matSize, int batchIdx)
{
    if (matSize <= 0)
        return {};
    return makeBlasMatrix(p.m, p.n, p.lda, p.aFill, static_cast<uint32_t>(batchIdx * 100 + 42));
}

static void GenerateBatchData(
    const AclblasSgeqrfBatchedParam& p, int effectiveBatch, int matSize, int tauSize,
    std::vector<std::vector<float>>& aBatch, std::vector<std::vector<float>>& tauBatch, std::vector<float*>& aPtrs,
    std::vector<float*>& tauPtrs)
{
    for (int b = 0; b < effectiveBatch; b++) {
        if (matSize > 0) {
            aBatch[b] = GenerateTestMatrix(p, matSize, b);
            aPtrs[b] = aBatch[b].empty() ? nullptr : aBatch[b].data();
        }
        if (tauSize > 0) {
            tauBatch[b].resize(static_cast<size_t>(tauSize), 0.0f);
            tauPtrs[b] = tauBatch[b].data();
        }
    }
}

static void VerifyBatchOutputs(
    const AclblasSgeqrfBatchedParam& p, const std::vector<float*>& aPtrs, const std::vector<float*>& tauPtrs,
    const std::vector<float*>& aGoldenPtrs, const std::vector<float*>& tauGoldenPtrs, int effectiveBatch, int matSize,
    int tauSize, const VerifyConfig& cfg)
{
    for (int b = 0; b < effectiveBatch; b++) {
        if (matSize > 0) {
            std::string batchCaseId = p.caseName + "_A_batch" + std::to_string(b);
            EXPECT_TRUE(
                Verifier::verifyVector(aPtrs[b], aGoldenPtrs[b], static_cast<size_t>(matSize), 1, cfg, batchCaseId));
        }
    }
    for (int b = 0; b < effectiveBatch; b++) {
        if (tauSize > 0) {
            std::string batchCaseId = p.caseName + "_Tau_batch" + std::to_string(b);
            EXPECT_TRUE(
                Verifier::verifyVector(
                    tauPtrs[b], tauGoldenPtrs[b], static_cast<size_t>(tauSize), 1, cfg, batchCaseId));
        }
    }
}

TEST_P(AclblasSgeqrfBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    const int effectiveBatch = std::max(0, p.batchSize);
    const int matSize = (p.m > 0 && p.n > 0) ? p.lda * p.n : 0;
    const int tauSize = (p.m > 0 && p.n > 0) ? std::min(p.m, p.n) : 0;

    std::vector<std::vector<float>> aBatch(effectiveBatch);
    std::vector<std::vector<float>> tauBatch(effectiveBatch);
    std::vector<float*> aPtrs(effectiveBatch, nullptr);
    std::vector<float*> tauPtrs(effectiveBatch, nullptr);
    GenerateBatchData(p, effectiveBatch, matSize, tauSize, aBatch, tauBatch, aPtrs, tauPtrs);

    float** aArrayPtr = p.aArrayNull ? nullptr : (aPtrs.empty() ? nullptr : aPtrs.data());
    float** tauArrayPtr = p.tauArrayNull ? nullptr : (tauPtrs.empty() ? nullptr : tauPtrs.data());
    std::vector<int> infoHost(std::max(1, effectiveBatch), 0);

    aclblasStatus_t ret = aclblasSgeqrfBatched_npu(
        AclblasSgeqrfBatchedArch35Test::handle_, p.m, p.n, reinterpret_cast<float* const*>(aArrayPtr), p.lda,
        reinterpret_cast<float* const*>(tauArrayPtr), infoHost.data(), p.batchSize);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.m == 0 || p.n == 0 || p.batchSize == 0)
        return;

    std::vector<std::vector<float>> aGoldenBatch(effectiveBatch);
    std::vector<std::vector<float>> tauGoldenBatch(effectiveBatch);
    std::vector<float*> aGoldenPtrs(effectiveBatch, nullptr);
    std::vector<float*> tauGoldenPtrs(effectiveBatch, nullptr);
    GenerateBatchData(p, effectiveBatch, matSize, tauSize, aGoldenBatch, tauGoldenBatch, aGoldenPtrs, tauGoldenPtrs);

    std::vector<int> infoGolden(std::max(1, effectiveBatch), 0);
    aclblasSgeqrfBatched_cpu(
        AclblasSgeqrfBatchedArch35Test::handle_, p.m, p.n, reinterpret_cast<float* const*>(aGoldenPtrs.data()), p.lda,
        reinterpret_cast<float* const*>(tauGoldenPtrs.data()), infoGolden.data(), p.batchSize);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold > 0.0 ? p.mereThreshold : (1.0 / 8192.0);
    cfg.mareMultiplier = p.mareMultiplier > 0.0 ? p.mareMultiplier : 10.0;

    VerifyBatchOutputs(p, aPtrs, tauPtrs, aGoldenPtrs, tauGoldenPtrs, effectiveBatch, matSize, tauSize, cfg);
}
