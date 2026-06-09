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
#include <algorithm>
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sscal_param.h"
#include "sscal_golden.h"
#include "sscal_npu_wrapper.h"

class SscalArch35Test : public BlasTest<SscalParam> {};

TEST_F(SscalArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float x[5] = {1, 2, 3, 4, 5};
    aclblasStatus_t ret = aclblasSscal_npu(nullptr, 5, &alpha, x, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Sscal, SscalArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SscalParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SscalParam>);

static std::vector<float> MakeSscalInput(const SscalParam& p)
{
    if (p.incx == 1) {
        return makeBlasArray(p.n, parseDataFill(p.xFill), p.xDesc, kBlasSentinel, p.randomSeed);
    }
    return makeBlasStrided(p.n, p.incx, parseDataFill(p.xFill), p.randomSeed);
}