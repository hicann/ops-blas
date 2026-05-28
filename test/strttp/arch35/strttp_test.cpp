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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "config.h"
#include "verify.h"
#include "gtest.h"
#include "strttp_golden.h"

static std::string g_configDir = ".";

namespace {

void freeDeviceBuffers(void* dA, void* dP) {
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dP != nullptr) {
        aclrtFree(dP);
    }
}

void fillStrttpRoundtripDense(uint32_t n, uint32_t lda, std::vector<float>& s) {
    s.assign(static_cast<size_t>(lda) * n, 0.0f);
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            s[j * lda + i] = static_cast<float>(j * n + i + 1);
        }
    }
}

}  // namespace

class StrttpTest : public ::testing::TestWithParam<TestCaseConfig> {
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

aclblasHandle_t StrttpTest::handle_ = nullptr;
aclrtStream StrttpTest::stream_ = nullptr;

static std::vector<TestCaseConfig> loadStrttpCases() {
    auto [cases, cfg] = ConfigLoader::loadAllForOp(g_configDir, "strttp");
    (void)cfg;
    for (auto& tc : cases) {
        tc.verifyCfg.mode = PrecisionMode::EXACT;
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Strttp, StrttpTest,
    ::testing::ValuesIn(loadStrttpCases()),
    gtestParamNameFromTestCase);

TEST_P(StrttpTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();
    const aclblasFillMode_t uplo = parseStrttpUplo(tc.uplo.value_or("LOWER"));
    const int n = static_cast<int>(tc.n.value_or(0));
    const int lda = static_cast<int>(tc.lda.value_or(n));

    if (!tc.expectSuccess) {
        if (tc.caseId == "TC_L0_01") {
            EXPECT_EQ(static_cast<int>(aclblasStrttp(nullptr, ACLBLAS_LOWER, 5, nullptr, 5, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
        } else if (tc.caseId == "TC_L0_02") {
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, -1, nullptr, 1, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        } else if (tc.caseId == "TC_L0_03") {
            std::vector<float> a(8);
            fillStrttpDenseMatrix(4, 2, STRTTP_NORMAL, a);
            void* dA = nullptr;
            void* dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 32, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&dP, 40, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(dA, 32, a.data(), 32, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, ACLBLAS_UPPER, 4,
                      static_cast<const float*>(dA), 2, static_cast<float*>(dP))),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            freeDeviceBuffers(dA, dP);
        } else if (tc.caseId == "TC_L0_04") {
            std::vector<float> a(16);
            fillStrttpDenseMatrix(4, 4, STRTTP_NORMAL, a);
            void* dA = nullptr;
            void* dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 64, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&dP, 40, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(dA, 64, a.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, static_cast<aclblasFillMode_t>(0xFF), 4,
                      static_cast<const float*>(dA), 4, static_cast<float*>(dP))),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            freeDeviceBuffers(dA, dP);
        } else if (tc.caseId == "TC_L1_31") {
            void* dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dP, 60, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 5, nullptr, 5,
                      static_cast<float*>(dP))),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(dP);
        } else if (tc.caseId == "TC_L1_32") {
            std::vector<float> a(25);
            fillStrttpDenseMatrix(5, 5, STRTTP_NORMAL, a);
            void* dA = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 100, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(dA, 100, a.data(), 100, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 5,
                      static_cast<const float*>(dA), 5, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(dA);
        } else if (tc.caseId == "TC_L1_33") {
            EXPECT_EQ(static_cast<int>(aclblasStrttp(nullptr, static_cast<aclblasFillMode_t>(0xFF), -1,
                      nullptr, 3, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
        } else if (tc.caseId == "TC_L1_34") {
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, static_cast<aclblasFillMode_t>(0xFF), 0,
                      nullptr, 1, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        } else if (tc.caseId == "TC_L1_36") {
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 5, nullptr, 5, nullptr)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        } else if (tc.caseId == "TC_L1_37") {
            std::vector<float> a(9);
            fillStrttpDenseMatrix(3, 3, STRTTP_NORMAL, a);
            void* dA = nullptr;
            void* dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 36, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&dP, 24, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(dA, 36, a.data(), 36, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, static_cast<aclblasFillMode_t>(120), 3,
                      static_cast<const float*>(dA), 3, static_cast<float*>(dP))),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            freeDeviceBuffers(dA, dP);
        } else if (tc.caseId == "TC_L1_38") {
            std::vector<float> a(9);
            fillStrttpDenseMatrix(3, 3, STRTTP_NORMAL, a);
            void* dA = nullptr;
            void* dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 36, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&dP, 24, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMemcpy(dA, 36, a.data(), 36, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, static_cast<aclblasFillMode_t>(123), 3,
                      static_cast<const float*>(dA), 3, static_cast<float*>(dP))),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            freeDeviceBuffers(dA, dP);
        }
        return;
    }

    if (n == 0) {
        EXPECT_EQ(static_cast<int>(aclblasStrttp(handle_, uplo, 0, nullptr, lda, nullptr)),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        return;
    }

    if (tc.caseId == "TC_L1_39" || tc.caseId == "TC_L1_40") {
        const uint32_t rn = 32;
        const uint32_t rlda = 32;
        const size_t rap = rn * (rn + 1) / 2;
        const size_t ral = rlda * rn;
        std::vector<float> s;
        fillStrttpRoundtripDense(rn, rlda, s);
        void* dS = nullptr;
        void* dP = nullptr;
        void* dD = nullptr;
        ASSERT_EQ(aclrtMalloc(&dS, ral * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dP, rap * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dD, ral * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(dS, ral * 4, s.data(), ral * 4, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        const aclblasFillMode_t rtUplo = (uplo == ACLBLAS_UPPER) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        ASSERT_EQ(static_cast<int>(aclblasStrttp(handle_, rtUplo, rn,
                      static_cast<const float*>(dS), rlda, static_cast<float*>(dP))),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(static_cast<int>(aclblasStpttr(handle_, rtUplo, rn,
                      static_cast<const float*>(dP), static_cast<float*>(dD), rlda)),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<float> r(ral);
        ASSERT_EQ(aclrtMemcpy(r.data(), ral * 4, dD, ral * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        if (rtUplo == ACLBLAS_LOWER) {
            for (uint32_t j = 0; j < rn; j++) {
                for (uint32_t i = j; i < rn; i++) {
                    ASSERT_EQ(r[j * rlda + i], s[j * rlda + i]);
                }
            }
        } else {
            for (uint32_t j = 0; j < rn; j++) {
                for (uint32_t i = 0; i <= j; i++) {
                    ASSERT_EQ(r[j * rlda + i], s[j * rlda + i]);
                }
            }
        }
        freeDeviceBuffers(dS, dP);
        aclrtFree(dD);
        return;
    }

    const StrttpSpecialValueType svt = strttpSpecialValueTypeFromDescription(tc.description);
    std::vector<float> aHost;
    fillStrttpDenseMatrix(n, lda, svt, aHost);
    const size_t apLen = static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2;
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    std::vector<float> apHost(apLen, 0.0f);

    void* dA = nullptr;
    void* dP = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, aLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dP, apLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dA, aLen * sizeof(float), aHost.data(), aLen * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dP, apLen * sizeof(float), apHost.data(), apLen * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    const aclblasStatus_t ret = aclblasStrttp(handle_, uplo, n,
                                              static_cast<const float*>(dA), lda,
                                              static_cast<float*>(dP));
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    ASSERT_EQ(aclrtMemcpy(apHost.data(), apLen * sizeof(float), dP, apLen * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    freeDeviceBuffers(dA, dP);

    std::vector<float> goldenOutput;
    strttp_golden_impl(tc, uplo, aHost, goldenOutput);
    EXPECT_TRUE(verifyDenseVector(tc, apHost, goldenOutput));
}

int main(int argc, char* argv[]) {
    g_configDir = (argc > 1) ? argv[1] : ".";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
