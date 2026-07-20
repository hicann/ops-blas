/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgemm_batched_param.h"
#include "cgemm_batched_golden.h"
#include "cgemm_batched_npu_wrapper.h"

struct CgemmBatchedTestData {
    std::vector<std::vector<aclblasComplex>> aBatch;
    std::vector<std::vector<aclblasComplex>> bBatch;
    std::vector<std::vector<aclblasComplex>> cBatch;
    std::vector<std::vector<aclblasComplex>> aGolden;
    std::vector<std::vector<aclblasComplex>> bGolden;
    std::vector<std::vector<aclblasComplex>> cGolden;
};

// Generate complex array from a float fill mode: use two independent float arrays for real/imag.
inline std::vector<aclblasComplex> makeComplexArray(int64_t size, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasArray(size, fill, seed);
    std::vector<float> imagPart = makeBlasArray(size, fill, seed + 7919);
    std::vector<aclblasComplex> data(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

inline CgemmBatchedTestData GenerateCgemmBatchedTestData(const CgemmBatchedParam& p)
{
    CgemmBatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physRowsA = cgemmBatchedPhysRows(p.m, p.k, p.transA);
    int physColsA = cgemmBatchedPhysCols(p.m, p.k, p.transA);
    int physRowsB = cgemmBatchedPhysRows(p.k, p.n, p.transB);
    int physColsB = cgemmBatchedPhysCols(p.k, p.n, p.transB);

    data.aBatch.resize(safeBatch);
    data.bBatch.resize(safeBatch);
    data.cBatch.resize(safeBatch);
    data.aGolden.resize(safeBatch);
    data.bGolden.resize(safeBatch);
    data.cGolden.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(b * 3);
        int64_t aSize = static_cast<int64_t>(std::max(1, p.lda)) * std::max(1, physColsA);
        int64_t bSize = static_cast<int64_t>(std::max(1, p.ldb)) * std::max(1, physColsB);
        int64_t cSize = static_cast<int64_t>(std::max(1, p.ldc)) * std::max(1, p.n);
        data.aBatch[b] = makeComplexArray(aSize, p.aFill, seed);
        data.bBatch[b] = makeComplexArray(bSize, p.bFill, seed + 1);
        data.cBatch[b] = makeComplexArray(cSize, p.cFill, seed + 2);
        data.aGolden[b] = data.aBatch[b];
        data.bGolden[b] = data.bBatch[b];
        data.cGolden[b] = data.cBatch[b];
    }
    return data;
}

struct CgemmBatchedPtrArrays {
    std::vector<const aclblasComplex*> aPtrs;
    std::vector<const aclblasComplex*> bPtrs;
    std::vector<aclblasComplex*> cPtrs;
};

class CgemmBatchedArch35Test : public BlasTest<CgemmBatchedParam> { };

TEST_F(CgemmBatchedArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f}, beta{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCgemmBatched(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8,
        &alpha, nullptr, 8, nullptr, 8,
        &beta, nullptr, 8, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    CgemmBatched, CgemmBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmBatchedParam>);

inline void VerifyCgemmBatchedPrecision(
    const CgemmBatchedParam& p, CgemmBatchedTestData& data,
    CgemmBatchedPtrArrays& ptrs, int safeBatch, size_t cCount,
    aclblasHandle_t testHandle)
{
    for (int b = 0; b < safeBatch; b++) {
        if (ptrs.cPtrs[b] == nullptr) continue;

        const aclblasComplex* const goldenA[] = { data.aGolden[b].data() };
        const aclblasComplex* const goldenB[] = { data.bGolden[b].data() };
        aclblasComplex* const goldenC[] = { data.cGolden[b].data() };

        aclblasCgemmBatched_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            &p.alpha, goldenA, p.lda, goldenB, p.ldb,
            &p.beta, goldenC, p.ldc, 1);

        std::string caseName = p.caseName + "_batch" + std::to_string(b);

        const aclblasComplex* out = data.cBatch[b].data();
        const aclblasComplex* gold = data.cGolden[b].data();
        // Split into real/imag and verify separately with mixed tolerance (FP32).
        std::vector<float> outR(cCount), outI(cCount), goldR(cCount), goldI(cCount);
        for (size_t i = 0; i < cCount; i++) {
            outR[i] = out[i].real; outI[i] = out[i].imag;
            goldR[i] = gold[i].real; goldI[i] = gold[i].imag;
        }
        VerifyConfig cfgR, cfgI;
        applyMixedTolerance(cfgR, ACL_FLOAT, goldR.data(), cCount);
        applyMixedTolerance(cfgI, ACL_FLOAT, goldI.data(), cCount);
        EXPECT_TRUE(Verifier::verifyVector(outR.data(), goldR.data(), cCount, 1, cfgR, caseName + "_real"));
        EXPECT_TRUE(Verifier::verifyVector(outI.data(), goldI.data(), cCount, 1, cfgI, caseName + "_imag"));
    }
}

TEST_P(CgemmBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = CgemmBatchedArch35Test::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateCgemmBatchedTestData(p);
    auto ptrs = BuildGemmBatchedPtrsTpl<CgemmBatchedTestData, CgemmBatchedParam, CgemmBatchedPtrArrays>(data, p, safeBatch);

    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &p.beta;
    const aclblasComplex* const* aPtrArr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrs.data();
    const aclblasComplex* const* bPtrArr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrs.data();
    aclblasComplex* const* cPtrArr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrs.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasCgemmBatched_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb,
            betaPtr, cPtrArr, p.ldc, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasCgemmBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb,
        betaPtr, cPtrArr, p.ldc, p.batchCount);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cPtrArr == nullptr) return;

    size_t cCount = static_cast<size_t>(std::max(1, p.ldc)) * p.n;
    VerifyCgemmBatchedPrecision(p, data, ptrs, safeBatch, cCount, testHandle);
}
