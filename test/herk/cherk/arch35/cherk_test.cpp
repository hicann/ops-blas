/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify_uplo.h"
#include "cherk_param.h"
#include "cherk_golden.h"
#include "cherk_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification helpers (测试方案 §7.3 / §7.4)
// ═══════════════════════════════════════════════════════════════════════════════

// Collect uplo / non-uplo element pairs from the NPU output and golden buffers.
// `isUplo` follows the BLAS uplo convention: UPPER ⇒ i <= j, LOWER ⇒ i >= j.
// `goldPtr` doubles as the "input old" reference for the non-uplo triangle
// because cblas_cherk only writes the uplo triangle.
template <typename Param>
static inline void CollectUploNonUploElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr,
    std::vector<float>& npuUploRe, std::vector<float>& npuUploIm,
    std::vector<float>& goldUploRe, std::vector<float>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm,
    std::vector<float>& oldNonRe, std::vector<float>& oldNonIm)
{
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUploRe.push_back(cPtr[idx].real);
                npuUploIm.push_back(cPtr[idx].imag);
                goldUploRe.push_back(goldPtr[idx].real);
                goldUploIm.push_back(goldPtr[idx].imag);
            } else {
                npuNonRe.push_back(cPtr[idx].real);
                npuNonIm.push_back(cPtr[idx].imag);
                oldNonRe.push_back(goldPtr[idx].real);
                oldNonIm.push_back(goldPtr[idx].imag);
            }
        }
    }
}

// Verify uplo triangle precision — real and imag parts use ACL_FLOAT mixed tolerance.
static inline void VerifyUploPrecision(
    const std::string& caseName,
    const float* npuUploRe, const float* goldUploRe, size_t uploCount,
    const float* npuUploIm, const float* goldUploIm)
{
    // uplo triangle — real part, ACL_FLOAT mixed tolerance
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldUploRe, uploCount);
    EXPECT_TRUE(Verifier::verifyVector(npuUploRe, goldUploRe, uploCount, 1, cfgRe,
        caseName + "_uplo_real"));

    // uplo triangle — imag part, ACL_FLOAT mixed tolerance
    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldUploIm, uploCount);
    EXPECT_TRUE(Verifier::verifyVector(npuUploIm, goldUploIm, uploCount, 1, cfgIm,
        caseName + "_uplo_imag"));
}

// Verify non-uplo triangle — must equal input old (EXACT, verifies no pollution).
static inline void VerifyNonUploExact(
    const std::string& caseName,
    const float* npuNonRe, const float* oldNonRe, size_t nonCount,
    const float* npuNonIm, const float* oldNonIm)
{
    if (nonCount == 0) {
        return;
    }
    VerifyConfig cfgNonRe;
    cfgNonRe.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonRe, oldNonRe, nonCount, 1, cfgNonRe,
        caseName + "_nonuplo_real"));
    VerifyConfig cfgNonIm;
    cfgNonIm.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonIm, oldNonIm, nonCount, 1, cfgNonIm,
        caseName + "_nonuplo_imag"));
}

// Verify Hermitian diagonal: when beta==0 output is pure A·A^H, diagonal must be real.
template <typename Param>
static inline void VerifyDiagonalHermitian(const Param& p, const aclblasComplex* cPtr)
{
    if (p.beta != 0.0f) {
        return;
    }
    double maxDiagImag = 0.0;
    for (int i = 0; i < p.n; i++) {
        size_t idx = static_cast<size_t>(i) + static_cast<size_t>(i) * p.ldc;
        double im = std::abs(static_cast<double>(cPtr[idx].imag));
        if (im > maxDiagImag) {
            maxDiagImag = im;
        }
    }
    // ACL_FLOAT atol = 2^-16 ≈ 1.5259e-5 (测试方案 §7.2)
    EXPECT_LE(maxDiagImag, 1.52587890625e-5)
        << "[" << p.caseName << "] Hermitian diagonal imag not zero (max=" << maxDiagImag << ")";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification (测试方案 §7.3 / §7.4)
//   1. uplo triangle: split real/imag, apply ACL_FLOAT mixed tolerance separately
//   2. non-uplo triangle: compare against the input-old C (EXACT, must be untouched)
//   3. Hermitian diagonal: |C[i][i].imag| <= atol when beta==0 (pure A·A^H output)
// `goldPtr` already carries the input-old values in its non-uplo triangle
// because cblas_cherk only writes the uplo triangle — so it doubles as the
// "input old" reference for the non-uplo check.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename Param>
static inline void VerifyUploTriangleComplex(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, size_t /*cCount*/)
{
    if (p.n <= 0) {
        return;
    }

    std::vector<float> npuUploRe, npuUploIm, goldUploRe, goldUploIm;
    std::vector<float> npuNonRe, npuNonIm, oldNonRe, oldNonIm;
    CollectUploNonUploElements(p, cPtr, goldPtr,
        npuUploRe, npuUploIm, goldUploRe, goldUploIm,
        npuNonRe, npuNonIm, oldNonRe, oldNonIm);

    VerifyUploPrecision(p.caseName,
        npuUploRe.data(), goldUploRe.data(), npuUploRe.size(),
        npuUploIm.data(), goldUploIm.data());

    VerifyNonUploExact(p.caseName,
        npuNonRe.data(), oldNonRe.data(), npuNonRe.size(),
        npuNonIm.data(), oldNonIm.data());

    VerifyDiagonalHermitian(p, cPtr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class CherkArch35Test : public BlasTest<CherkParam> {
};

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(CherkArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasCherk_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4,
        &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Cherk, CherkArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CherkParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CherkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct CherkHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> cGolden;
    const aclblasComplex* aPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const CherkParam& p, CherkHostData& d)
{
    const int aRows = p.lda;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    // 3 copies of C (npu output + golden + input-old reference) + A.
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL; // §8 遗留问题 4
    if (3 * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (3 * cBytes + aBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        d.aHost = makeBlasComplexMatrix(aRows, aCols, p.lda, p.aFill, p.randomSeed);
        d.cHost = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 1U);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);

    if (d.cPtr != nullptr) {
        try {
            d.cGolden = d.cHost; // golden starts from input-old C; non-uplo triangle stays untouched
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
    }
    return true;
}

TEST_P(CherkArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    CherkHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    // nullAlpha: wrapper forwards nullptr for alpha so the API's INVALID_VALUE
    // path is exercised. The alpha value in p.alpha is irrelevant in that case.
    aclblasStatus_t ret = aclblasCherk_npu(
        CherkArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc, p.nullAlpha);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    aclblasStatus_t goldenRet = aclblasCherk_cpu(
        CherkArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    VerifyUploTriangleComplex(p, d.cPtr, d.cGoldenPtr, d.cCount);
}
