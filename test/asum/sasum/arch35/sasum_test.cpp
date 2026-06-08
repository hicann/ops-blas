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
#include "sasum_param.h"
#include "sasum_golden.h"
#include "sasum_npu_wrapper.h"

class SasumArch35Test : public BlasTest<SasumParam> {};

TEST_F(SasumArch35Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasSasum_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Sasum, SasumArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SasumParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SasumParam>);

TEST_P(SasumArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    std::cout << "\n=== RUN  [" << p.caseName << "] n=" << p.n << " incx=" << p.incx
              << " x_method=" << static_cast<int>(p.x.method) << " desc=" << p.description << " ===" << std::endl;

    const int64_t xLen = (p.n > 0) ? static_cast<int64_t>(1 + (p.n - 1) * p.incx) : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 0.0f;
    aclblasStatus_t ret = aclblasSasum_npu(SasumArch35Test::handle_, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] unexpected return code: got " << static_cast<int>(ret) << " expect "
        << static_cast<int>(p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    float golden = 0.0f;
    aclblasSasum_cpu(SasumArch35Test::handle_, p.n, xHost.data(), p.incx, &golden);

    bool outputIsInf = std::isinf(result);
    bool goldenIsInf = std::isinf(golden);
    if (outputIsInf || goldenIsInf) {
        EXPECT_TRUE(outputIsInf && goldenIsInf)
            << "[" << p.caseName << "] overflow mismatch: output=" << result << " golden=" << golden;
        return;
    }

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::ABS;
    cfg.absTol = p.absTol;
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
