/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "config.h"
#include "device.h"
#include "gtest.h"
#include "gbmv_test_utils.h"
#include "gbmv_golden.h"

static std::string g_configDir = ".";

namespace {

struct GbmvKernelArgs {
    const float* a = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    int incx = 1;
    int incy = 1;
};

GbmvKernelArgs makeGbmvKernelArgs(void* dA, void* dX, void* dY,
                                  int64_t incx, int64_t incy) {
    return {
        static_cast<const float*>(dA),
        static_cast<const float*>(dX),
        static_cast<float*>(dY),
        static_cast<int>(incx),
        static_cast<int>(incy),
    };
}

}  // namespace

class GbmvTest : public ::testing::TestWithParam<TestCaseConfig> {
protected:
    static void SetUpTestSuite() {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclblasCreate(&handle_);
        aclrtCreateStream(&stream_);
        aclblasSetStream(handle_, stream_);
    }
    static void TearDownTestSuite() {
        aclrtDestroyStream(stream_);
        aclblasDestroy(handle_);
        aclrtResetDevice(0);
        aclFinalize();
    }

    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

aclblasHandle_t GbmvTest::handle_ = nullptr;
aclrtStream GbmvTest::stream_ = nullptr;

static std::vector<TestCaseConfig> loadGbmvCases() {
    auto [cases, cfg] = ConfigLoader::loadAllForOp(g_configDir, "gbmv");
    (void)cfg;
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Gbmv, GbmvTest,
    ::testing::ValuesIn(loadGbmvCases()),
    gtestParamNameFromTestCase);

TEST_P(GbmvTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();

    auto hostData = generateHostDataGbmv(tc);
    auto devBufs = allocAndCopyToDevice({hostData[0], hostData[1], hostData[2]});

    const float alpha = tc.alphaReal.value_or(1.0f);
    const float beta = tc.betaReal.value_or(0.0f);
    const aclblasOperation_t transOp = parseTrans(tc.trans.value_or("N"));
    const GbmvKernelArgs kernelArgs = makeGbmvKernelArgs(
        devBufs[0]->ptr(), devBufs[1]->ptr(), devBufs[2]->ptr(),
        tc.incx.value_or(1), tc.incy.value_or(1));

    const aclblasStatus_t ret = aclblasSgbmv(
        handle_, transOp,
        static_cast<int>(tc.m.value_or(0)),
        static_cast<int>(tc.n.value_or(0)),
        static_cast<int>(tc.kl.value_or(0)),
        static_cast<int>(tc.ku.value_or(0)),
        &alpha, kernelArgs.a,
        static_cast<int>(tc.lda.value_or(0)),
        kernelArgs.x, kernelArgs.incx,
        &beta, kernelArgs.y, kernelArgs.incy);

    if (!tc.expectSuccess) {
        EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    std::vector<float> outputHost = hostData[2];
    devBufs[2]->copyToHost(outputHost.data(), outputHost.size() * sizeof(float));

    std::vector<float> goldenOutput;
    gbmv_golden_impl(tc, hostData, goldenOutput);

    EXPECT_TRUE(verifyGbmvResult(tc, outputHost, goldenOutput));
}

int main(int argc, char *argv[]) {
    g_configDir = (argc > 1) ? argv[1] : ".";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
