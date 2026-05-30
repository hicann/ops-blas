/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cann_ops_blas.h"
#include "config.h"
#include "device.h"
#include "srotm_golden.h"
#include "srotm_test_utils.h"

static std::string getSrotmConfigDir()
{
    const std::string filePath = __FILE__;
    const size_t pos = filePath.find_last_of("/\\");
    return pos == std::string::npos ? "." : filePath.substr(0, pos);
}

static std::string gtestParamNameFromTestCase(const ::testing::TestParamInfo<TestCaseConfig>& info)
{
    std::string name = info.param.caseId;
    for (auto& c : name) {
        if (c == '-') {
            c = '_';
        }
    }
    return name;
}

class SrotmTest : public ::testing::TestWithParam<TestCaseConfig> {
protected:
    static void SetUpTestSuite()
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclblasCreate(&handle_);
        aclrtCreateStream(&stream_);
        aclblasSetStream(handle_, stream_);
    }

    static void TearDownTestSuite()
    {
        aclrtDestroyStream(stream_);
        aclblasDestroy(handle_);
        aclrtResetDevice(0);
        aclFinalize();
    }

    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

aclblasHandle_t SrotmTest::handle_ = nullptr;
aclrtStream SrotmTest::stream_ = nullptr;

static std::vector<TestCaseConfig> loadSrotmCases()
{
    auto [cases, cfg] = ConfigLoader::loadAllForOp(getSrotmConfigDir(), "srotm");
    (void)cfg;
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Srotm, SrotmTest,
    ::testing::ValuesIn(loadSrotmCases()),
    gtestParamNameFromTestCase);

TEST_P(SrotmTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();

    auto hostData = generateHostDataSrotm(tc);
    auto devBufs = allocAndCopyToDevice(hostData);

    std::array<float, 5> sparam = getSrotmParams(tc);
    const int64_t n = tc.n.value_or(0);
    const int64_t incx = tc.incx.value_or(1);
    const int64_t incy = tc.incy.value_or(1);

    const aclblasStatus_t ret = aclblasSrotm(
        handle_, devBufs[0]->floatPtr(), devBufs[1]->floatPtr(), sparam.data(), n, incx, incy);

    if (!tc.expectSuccess) {
        EXPECT_NE(ret, ACLBLAS_STATUS_SUCCESS);
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    std::vector<float> outputX = hostData[0];
    std::vector<float> outputY = hostData[1];
    devBufs[0]->copyToHost(outputX.data(), outputX.size() * sizeof(float));
    devBufs[1]->copyToHost(outputY.data(), outputY.size() * sizeof(float));

    std::vector<float> goldenX;
    std::vector<float> goldenY;
    srotm_golden_impl(tc, hostData, goldenX, goldenY);

    EXPECT_TRUE(verifySrotmResult(tc, outputX, goldenX));
    EXPECT_TRUE(verifySrotmResult(tc, outputY, goldenY));
}
