/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "chemm_param.h"
#include "chemm_golden.h"
#include "chemm_npu_wrapper.h"

class ChemmArch22Test : public BlasTest<ChemmParam> {};

constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024;

inline int64_t ChemmSafeElemCount(int64_t rows, int64_t cols)
{
    if (cols == 0 || rows <= 0) {
        return 0;
    }
    if (rows > kMaxSafeTestElems / cols) {
        return 0;
    }
    return rows * cols;
}

inline void ChemmFixDiagonal(std::vector<float>& values, const ChemmParam& p, int64_t aDim)
{
    if (p.m <= 0 || p.n <= 0) {
        return;
    }
    int64_t diagonal = std::min(aDim, p.lda);
    for (int64_t d = 0; d < diagonal; ++d) {
        size_t index = 2 * static_cast<size_t>(d * p.lda + d);
        if (index + 1 < values.size()) {
            values[index + 1] = 0.0f;
        }
    }
}

inline void ChemmPrepareHostData(const ChemmParam& p, std::vector<float>& a, std::vector<float>& b,
    std::vector<float>& c, std::vector<float>& result)
{
    int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    int64_t aCount = ChemmSafeElemCount(aDim, p.lda);
    int64_t bCount = ChemmSafeElemCount(p.m, p.ldb);
    int64_t cCount = ChemmSafeElemCount(p.m, p.ldc);
    if (aCount > 0 && p.aFill.method != BlasFillMode::M_NULLPTR) {
        a = makeBlasArray(2 * aCount, p.aFill, p.randomSeed);
        ChemmFixDiagonal(a, p, aDim);
    }
    if (bCount > 0 && p.bFill.method != BlasFillMode::M_NULLPTR) {
        b = makeBlasArray(2 * bCount, p.bFill, p.randomSeed + 1);
    }
    if (cCount > 0 && p.cFill.method != BlasFillMode::M_NULLPTR) {
        c = makeBlasArray(2 * cCount, p.cFill, p.randomSeed + 2);
        result = c;
    }
}

inline void ChemmBuildPointers(const ChemmParam& p, std::vector<float>& a, std::vector<float>& b,
    std::vector<float>& result, aclblasComplex& alpha, aclblasComplex& beta,
    const aclblasComplex*& alphaPtr, const aclblasComplex*& betaPtr,
    const aclblasComplex*& aPtr, const aclblasComplex*& bPtr, aclblasComplex*& cPtr)
{
    alpha = {p.alphaReal, p.alphaImag};
    beta = {p.betaReal, p.betaImag};
    alphaPtr = p.nullAlpha ? nullptr : &alpha;
    betaPtr = p.nullBeta ? nullptr : &beta;
    aPtr = a.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(a.data());
    bPtr = b.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(b.data());
    cPtr = result.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(result.data());
}

inline bool ChemmCheckExpectedStatus(const ChemmParam& p, aclblasStatus_t ret)
{
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS) {
        return true;
    }
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    return false;
}

inline void ChemmBuildGolden(const ChemmParam& p, const std::vector<float>& cHost,
    const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* a,
    const aclblasComplex* b, std::vector<float>& golden, aclblasHandle_t handle)
{
    std::string goldenFile = makeGoldenDir(std::string(__FILE__)) + p.caseName + ".golden";
    uint64_t hash = chemm_golden_cache::computeParamHash(p);
    if (chemm_golden_cache::loadGolden(goldenFile, golden, hash)) {
        return;
    }
    golden = cHost;
    aclblasComplex* output = golden.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(golden.data());
    aclblasChemm_cpu(handle, p.side, p.uplo, p.m, p.n, alpha, a, p.lda, b, p.ldb, beta, output, p.ldc);
    chemm_golden_cache::saveGolden(goldenFile, golden, hash);
}

inline void ChemmVerifyResult(const ChemmParam& p, const std::vector<float>& result,
    const std::vector<float>& golden)
{
    size_t count = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc) * 2;
    if (count == 0) {
        return;
    }
    VerifyConfig config;
    config.mode = PrecisionMode::MERE_MARE;
    config.mereThreshold = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
    config.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
    EXPECT_TRUE(Verifier::verifyVector(result.data(), golden.data(), count, 1, config, p.caseName));
}

// null handle: TEST_F (not CSV-driven), per blas-ST-develop convention
TEST_F(ChemmArch22Test, NullHandle)
{
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.5f, 0.0f};
    // Minimal dummy buffers to satisfy non-null pointer check; handle=nullptr
    // is caught before any pointer deref.
    std::vector<float> aBuf(32, 0.0f); // 4*4*2 floats
    std::vector<float> bBuf(32, 0.0f);
    std::vector<float> cBuf(32, 0.0f);
    aclblasStatus_t ret = aclblasChemm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, reinterpret_cast<aclblasComplex*>(aBuf.data()), 4,
        reinterpret_cast<aclblasComplex*>(bBuf.data()), 4, &beta, reinterpret_cast<aclblasComplex*>(cBuf.data()), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Chemm, ChemmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<ChemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChemmParam>);

TEST_P(ChemmArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    std::vector<float> aHostFlt;
    std::vector<float> bHostFlt;
    std::vector<float> cHostFlt;
    std::vector<float> cResultFlt;

    ChemmPrepareHostData(p, aHostFlt, bHostFlt, cHostFlt, cResultFlt);
    aclblasComplex alphaVal;
    aclblasComplex betaVal;
    const aclblasComplex* alphaPtr;
    const aclblasComplex* betaPtr;
    const aclblasComplex* aPtr;
    const aclblasComplex* bPtr;
    aclblasComplex* cPtr;
    ChemmBuildPointers(p, aHostFlt, bHostFlt, cResultFlt, alphaVal, betaVal,
        alphaPtr, betaPtr, aPtr, bPtr, cPtr);
    aclblasStatus_t ret = aclblasChemm_npu(
        ChemmArch22Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (!ChemmCheckExpectedStatus(p, ret)) {
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    std::vector<float> cGoldenFlt;
    ChemmBuildGolden(p, cHostFlt, alphaPtr, betaPtr, aPtr, bPtr, cGoldenFlt, ChemmArch22Test::handle_);
    ChemmVerifyResult(p, cResultFlt, cGoldenFlt);
}
