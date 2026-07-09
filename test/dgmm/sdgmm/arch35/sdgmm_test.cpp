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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sdgmm_param.h"
#include "sdgmm_golden.h"
#include "sdgmm_npu_wrapper.h"

// ── Test fixture ─────────────────────────────────────────────────────────────
class SdgmmArch35Test : public BlasTest<SdgmmParam> { };

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(SdgmmArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasSdgmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, 4, 4,
        nullptr, 4, nullptr, 1, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Sdgmm, SdgmmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SdgmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SdgmmParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(SdgmmArch35Test, CsvDriven) {
    const auto& p = GetParam();

    // Step 1: Generate host data
    // x length: mode=L -> m, mode=R -> n
    const int xLen = (p.mode == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    std::vector<float> xHost;
    if (p.nullx == 0 && p.m > 0 && p.n > 0) {
        // MIXED_RANDOM_INF / MIXED_RANDOM_NAN fills (first half RANDOM, second half
        // special value) need the dedicated mixed generator; everything else uses
        // the standard strided generator.
        const std::string& xRaw = p.xFillRaw;
        if (xRaw.rfind("MIXED", 0) == 0) {
            xHost = makeBlasMixed(xLen, p.incx, xRaw, p.randomSeed);
        } else {
            xHost = makeBlasStrided(xLen, p.incx, p.xFill, p.randomSeed);
        }
    }

    std::vector<float> aHost;
    if (p.nullA == 0 && p.m > 0 && p.n > 0) {
        aHost = makeBlasMatrix(p.m, p.n, p.lda, p.aFill, p.randomSeed);
    }

    // C is the output buffer; initialise with sentinel so unmodified padding
    // (if any) matches the golden's untouched region.
    std::vector<float> cHost;
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), kBlasSentinel);
    }

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    float*       cPtr = cHost.empty() ? nullptr : cHost.data();

    // Step 2: Execute on NPU (wrapper handles nullptr passthrough, device memory)
    aclblasStatus_t ret = aclblasSdgmm_npu(
        SdgmmArch35Test::handle_, p.mode, p.m, p.n,
        aPtr, p.lda, xPtr, p.incx, cPtr, p.ldc);

    // Step 3: Verify expected return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    // m==0 or n==0: operator returns SUCCESS without computing; no output to verify
    if (p.m == 0 || p.n == 0) return;

    // Step 4: Compute golden on CPU
    // xHost and aHost are not modified by the NPU wrapper (only cHost is written
    // via D2H), so they remain valid for golden computation.
    std::vector<float> goldenC(cHost.size(), kBlasSentinel);
    aclblasStatus_t cpuRet = aclblasSdgmm_cpu(
        SdgmmArch35Test::handle_, p.mode, p.m, p.n,
        aPtr, p.lda, xPtr, p.incx, goldenC.data(), p.ldc);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification — MERE_MARE (FP32: MERE < 2^-13, MARE < 10*2^-13)
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;

    // Compare entire C storage (ldc * n elements, stride 1).
    // Padding rows (if lda/ldc > m) are sentinel in both cHost and goldenC.
    EXPECT_TRUE(Verifier::verifyVector(
        cPtr, goldenC.data(), cHost.size(), 1, cfg, p.caseName));
}
