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
#include <string>

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "config.h"
#include "verify.h"
#include "gtest.h"
#include "stpttr_golden.h"

// Set in main() before RUN_ALL_TESTS(); ConfigLoader reads CSV/JSON from this dir.
static std::string g_configDir = ".";

namespace {

// Release packed AP (dP) and dense output A (dA) device buffers after a test case.
void freeDeviceBuffers(void* dP, void* dA) {
    if (dP != nullptr) {
        aclrtFree(dP);
    }
    if (dA != nullptr) {
        aclrtFree(dA);
    }
}

}  // namespace

// CSV-driven ST for aclblasStpttr (packed triangular AP -> dense triangular A).
// One handle/stream for the whole suite; aclInit/aclFinalize must not run per case.
class StpttrTest : public ::testing::TestWithParam<TestCaseConfig> {
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

aclblasHandle_t StpttrTest::handle_ = nullptr;
aclrtStream StpttrTest::stream_ = nullptr;

// Loaded when GTest registers parameters; g_configDir is already set in main().
static std::vector<TestCaseConfig> loadStpttrCases() {
    auto [cases, cfg] = ConfigLoader::loadAllForOp(g_configDir, "stpttr");
    (void)cfg;
    for (auto& tc : cases) {
        tc.verifyCfg.mode = PrecisionMode::EXACT;
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Stpttr, StpttrTest,
    ::testing::ValuesIn(loadStpttrCases()),
    gtestParamNameFromTestCase);

TEST_P(StpttrTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();
    const aclblasFillMode_t uplo = parseUplo(tc.uplo.value_or("LOWER"));
    const int n = static_cast<int>(tc.n.value_or(0));
    const int lda = static_cast<int>(tc.lda.value_or(n));

    // expect_success=false: each caseId checks a specific error return (no golden).
    if (!tc.expectSuccess) {
        if (tc.caseId == "TC_L0_01") {
            // Uninitialized handle
            EXPECT_EQ(static_cast<int>(aclblasStpttr(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5)),
                      static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
        } else if (tc.caseId == "TC_L0_02") {
            // Invalid n < 0
            void *d1 = nullptr, *d2 = nullptr;
            ASSERT_EQ(aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, -1, (const float*)d1, (float*)d2, 1)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(d1);
            aclrtFree(d2);
        } else if (tc.caseId == "TC_L0_03") {
            // lda too small for n
            void *dP = nullptr, *dA = nullptr;
            ASSERT_EQ(aclrtMalloc(&dP, 60, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&dA, 60, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, (const float*)dP, (float*)dA, 3)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(dP);
            aclrtFree(dA);
        } else if (tc.caseId == "TC_L0_04") {
            // Invalid uplo enum
            void *d1 = nullptr, *d2 = nullptr;
            ASSERT_EQ(aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)0xFF, 5, (const float*)d1, (float*)d2, 5)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(d1);
            aclrtFree(d2);
        } else if (tc.caseId == "TC_L1_31") {
            // AP pointer is nullptr
            void *dA = nullptr;
            ASSERT_EQ(aclrtMalloc(&dA, 100, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, nullptr, (float*)dA, 5)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(dA);
        } else if (tc.caseId == "TC_L1_32") {
            // A pointer is nullptr
            void *dP = nullptr;
            ASSERT_EQ(aclrtMalloc(&dP, 60, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, (const float*)dP, nullptr, 5)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(dP);
        } else if (tc.caseId == "TC_L1_33") {
            // Uninitialized handle with other invalid args
            EXPECT_EQ(static_cast<int>(aclblasStpttr(nullptr, (aclblasFillMode_t)0xFF, -1, nullptr, nullptr, 3)),
                      static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
        } else if (tc.caseId == "TC_L1_34") {
            // Invalid uplo with n=0
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)0xFF, 0, nullptr, nullptr, 1)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        } else if (tc.caseId == "TC_L1_36") {
            // Both AP and A are nullptr (n > 0)
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, nullptr, nullptr, 5)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        } else if (tc.caseId == "TC_L1_37") {
            // uplo below valid range (121=UPPER, 122=LOWER)
            void *d1 = nullptr, *d2 = nullptr;
            ASSERT_EQ(aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, static_cast<aclblasFillMode_t>(120), 3,
                      (const float*)d1, (float*)d2, 3)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(d1);
            aclrtFree(d2);
        } else if (tc.caseId == "TC_L1_38") {
            // uplo above valid range
            void *d1 = nullptr, *d2 = nullptr;
            ASSERT_EQ(aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            ASSERT_EQ(aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
            EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, static_cast<aclblasFillMode_t>(123), 3,
                      (const float*)d1, (float*)d2, 3)),
                      static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
            aclrtFree(d1);
            aclrtFree(d2);
        }
        return;
    }

    // n=0 with expect_success=true: early return SUCCESS; nullptr buffers are valid.
    if (n == 0) {
        EXPECT_EQ(static_cast<int>(aclblasStpttr(handle_, uplo, 0, nullptr, nullptr, lda)),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        return;
    }

    // Roundtrip: dense S -> aclblasStrttp -> packed P -> aclblasStpttr -> dense D; compare D vs S.
    if (tc.caseId == "TC_L1_39" || tc.caseId == "TC_L1_40") {
        const uint32_t rn = 32;
        const uint32_t rlda = 32;
        const size_t rap = rn * (rn + 1) / 2;
        const size_t ral = rlda * rn;
        std::vector<float> s(ral);
        for (uint32_t j = 0; j < rn; j++) {
            for (uint32_t i = 0; i < rn; i++) {
                s[j * rlda + i] = static_cast<float>(j * rn + i + 1);
            }
        }
        void *dS = nullptr, *dP = nullptr, *dD = nullptr;
        ASSERT_EQ(aclrtMalloc(&dS, ral * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dP, rap * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dD, ral * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(dS, ral * 4, s.data(), ral * 4, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        const aclblasFillMode_t rtUplo = (uplo == ACLBLAS_UPPER) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        ASSERT_EQ(static_cast<int>(aclblasStrttp(handle_, rtUplo, rn, (const float*)dS, rlda, (float*)dP)),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(static_cast<int>(aclblasStpttr(handle_, rtUplo, rn, (const float*)dP, (float*)dD, rlda)),
                  static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<float> r(ral);
        ASSERT_EQ(aclrtMemcpy(r.data(), ral * 4, dD, ral * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        // Only the stored triangle must match; the complementary part is undefined.
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
        aclrtFree(dS);
        aclrtFree(dP);
        aclrtFree(dD);
        return;
    }

    // Normal path: kernel reads/writes device memory directly.
    const StpttrSpecialValueType svt = specialValueTypeFromDescription(tc.description);
    const std::vector<float> ap = makeStpttrApData(n, svt);  // packed AP, size n*(n+1)/2
    const int aLen = lda * n;
    std::vector<float> aHost(aLen, kStpttrSentinel);  // untouched entries stay sentinel (-999)

    void* dP = nullptr;  // packed input AP on device
    void* dA = nullptr;  // dense output A on device (column-major, lda stride)
    ASSERT_EQ(aclrtMalloc(&dP, ap.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dA, static_cast<size_t>(aLen) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dP, ap.size() * sizeof(float), ap.data(), ap.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dA, static_cast<size_t>(aLen) * sizeof(float), aHost.data(),
                          static_cast<size_t>(aLen) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    const aclblasStatus_t ret = aclblasStpttr(handle_, uplo, n, static_cast<const float*>(dP),
                                              static_cast<float*>(dA), lda);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    ASSERT_EQ(aclrtMemcpy(aHost.data(), static_cast<size_t>(aLen) * sizeof(float), dA,
                          static_cast<size_t>(aLen) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    freeDeviceBuffers(dP, dA);

    // Golden + EXACT verify (only the active triangle; sentinel elsewhere).
    std::vector<float> goldenOutput;
    stpttr_golden_impl(tc, uplo, ap, goldenOutput);
    EXPECT_TRUE(verifyDenseVector(tc, aHost, goldenOutput));
}

// Custom main: argv[1] is the config dir (build.sh passes build/test/stpttr).
int main(int argc, char* argv[]) {
    g_configDir = (argc > 1) ? argv[1] : ".";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
