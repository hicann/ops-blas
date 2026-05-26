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
#include <cmath>
#include <cfloat>
#include <vector>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"

constexpr float kSentinel = -999.0f;

class StpttrTest : public ::testing::Test {
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

    static void Golden(int n, aclblasFillMode_t uplo, const float* ap, int lda, float* a)
    {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < lda; i++)
                a[j * lda + i] = kSentinel;
        int idx = 0;
        if (uplo == ACLBLAS_LOWER) {
            for (int j = 0; j < n; j++)
                for (int i = j; i < n; i++)
                    a[j * lda + i] = ap[idx++];
        } else {
            for (int j = 0; j < n; j++)
                for (int i = 0; i <= j; i++)
                    a[j * lda + i] = ap[idx++];
        }
    }

    bool Run(int n, int lda, aclblasFillMode_t uplo)
    {
        std::vector<float> ap(n * (n + 1) / 2);
        for (size_t i = 0; i < ap.size(); i++)
            ap[i] = static_cast<float>(i + 1);
        return Run(n, lda, uplo, ap);
    }

    bool Run(int n, int lda, aclblasFillMode_t uplo, const std::vector<float>& ap)
    {
        int aLen = lda * n;
        std::vector<float> a(aLen, kSentinel), gld(aLen);
        Golden(n, uplo, ap.data(), lda, gld.data());

        void *dP = nullptr, *dA = nullptr;
        aclError aclRet = aclrtMalloc(&dP, ap.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS)
            return false;
        aclRet = aclrtMalloc(&dA, aLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dP);
            return false;
        }

        aclRet =
            aclrtMemcpy(dP, ap.size() * sizeof(float), ap.data(), ap.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dP);
            aclrtFree(dA);
            return false;
        }
        aclRet = aclrtMemcpy(dA, aLen * sizeof(float), a.data(), aLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dP);
            aclrtFree(dA);
            return false;
        }

        auto ret = aclblasStpttr(handle_, uplo, n, static_cast<const float*>(dP), static_cast<float*>(dA), lda);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            aclrtFree(dP);
            aclrtFree(dA);
            return false;
        }
        aclrtSynchronizeStream(stream_);

        aclRet = aclrtMemcpy(a.data(), aLen * sizeof(float), dA, aLen * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dP);
        aclrtFree(dA);
        if (aclRet != ACL_SUCCESS)
            return false;

        for (int i = 0; i < aLen; i++) {
            if (a[i] != a[i] && gld[i] != gld[i])
                continue;
            if (a[i] != gld[i])
                return false;
        }
        return true;
    }
};

aclblasHandle_t StpttrTest::handle_ = nullptr;
aclrtStream StpttrTest::stream_ = nullptr;

// L0 — parameter validation
TEST_F(StpttrTest, L0_01_handle_null)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5)),
        static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}
TEST_F(StpttrTest, L0_02_n_negative)
{
    void *d1, *d2;
    aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, -1, (const float*)d1, (float*)d2, 1)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(d1);
    aclrtFree(d2);
}
TEST_F(StpttrTest, L0_03_lda_small)
{
    void *dP, *dA;
    aclrtMalloc(&dP, 60, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dA, 60, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, (const float*)dP, (float*)dA, 3)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dP);
    aclrtFree(dA);
}
TEST_F(StpttrTest, L0_04_uplo_bad)
{
    void *d1, *d2;
    aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)0xFF, 5, (const float*)d1, (float*)d2, 5)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(d1);
    aclrtFree(d2);
}
TEST_F(StpttrTest, L0_05_n_zero)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 0, nullptr, nullptr, 1)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

// L0 — normal shapes
TEST_F(StpttrTest, L0_06_n1_lower) { EXPECT_TRUE(Run(1, 1, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_07_n1_upper) { EXPECT_TRUE(Run(1, 1, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L0_08_n2_lower) { EXPECT_TRUE(Run(2, 2, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_09_n2_upper) { EXPECT_TRUE(Run(2, 2, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L0_10_n4_lower) { EXPECT_TRUE(Run(4, 4, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_11_n4_upper) { EXPECT_TRUE(Run(4, 4, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L0_12_n32_lower) { EXPECT_TRUE(Run(32, 32, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_13_n32_upper) { EXPECT_TRUE(Run(32, 32, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L0_14_n128_lower) { EXPECT_TRUE(Run(128, 128, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_15_n128_upper) { EXPECT_TRUE(Run(128, 128, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L0_16_n512_lower) { EXPECT_TRUE(Run(512, 512, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L0_17_n512_upper) { EXPECT_TRUE(Run(512, 512, ACLBLAS_UPPER)); }

// L1 — extended shapes
TEST_F(StpttrTest, L1_01_n8_lower) { EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_02_n8_upper) { EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_03_n9_lower) { EXPECT_TRUE(Run(9, 9, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_04_n9_upper) { EXPECT_TRUE(Run(9, 9, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_05_n64_lower) { EXPECT_TRUE(Run(64, 64, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_06_n64_upper) { EXPECT_TRUE(Run(64, 64, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_07_n100_lower) { EXPECT_TRUE(Run(100, 100, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_08_n100_upper) { EXPECT_TRUE(Run(100, 100, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_09_n256_lower) { EXPECT_TRUE(Run(256, 256, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_10_n256_upper) { EXPECT_TRUE(Run(256, 256, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_11_n500_lower) { EXPECT_TRUE(Run(500, 500, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_12_n500_upper) { EXPECT_TRUE(Run(500, 500, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_13_n1024_lower) { EXPECT_TRUE(Run(1024, 1024, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_14_n1024_upper) { EXPECT_TRUE(Run(1024, 1024, ACLBLAS_UPPER)); }

// L1 — non-compact lda
TEST_F(StpttrTest, L1_15_lda12_lower) { EXPECT_TRUE(Run(8, 12, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_16_lda12_upper) { EXPECT_TRUE(Run(8, 12, ACLBLAS_UPPER)); }
TEST_F(StpttrTest, L1_17_lda32_lower) { EXPECT_TRUE(Run(16, 32, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_18_lda32_upper) { EXPECT_TRUE(Run(16, 32, ACLBLAS_UPPER)); }

// L1 — special values
TEST_F(StpttrTest, L1_19_zeros_lower)
{
    std::vector<float> ap(36, 0);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_20_zeros_upper)
{
    std::vector<float> ap(36, 0);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}
TEST_F(StpttrTest, L1_21_large_lower)
{
    std::vector<float> ap(36, 1e10f);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_22_large_upper)
{
    std::vector<float> ap(36, 1e10f);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}
TEST_F(StpttrTest, L1_23_neg_lower)
{
    std::vector<float> ap(36);
    for (int i = 0; i < 36; i++)
        ap[i] = -static_cast<float>(i + 1);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_24_neg_upper)
{
    std::vector<float> ap(36);
    for (int i = 0; i < 36; i++)
        ap[i] = -static_cast<float>(i + 1);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}
TEST_F(StpttrTest, L1_25_inf_lower)
{
    std::vector<float> ap(36, INFINITY);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_26_inf_upper)
{
    std::vector<float> ap(36, INFINITY);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}
TEST_F(StpttrTest, L1_27_nan_lower)
{
    std::vector<float> ap(36, NAN);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_28_nan_upper)
{
    std::vector<float> ap(36, NAN);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}
TEST_F(StpttrTest, L1_29_extr_lower)
{
    std::vector<float> ap(36);
    float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (int i = 0; i < 36; i++)
        ap[i] = v[i % 7];
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, ap));
}
TEST_F(StpttrTest, L1_30_extr_upper)
{
    std::vector<float> ap(36);
    float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (int i = 0; i < 36; i++)
        ap[i] = v[i % 7];
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, ap));
}

// L1 — null pointers
TEST_F(StpttrTest, L1_31_ap_null)
{
    void* dA;
    aclrtMalloc(&dA, 100, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, nullptr, (float*)dA, 5)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
}
TEST_F(StpttrTest, L1_32_a_null)
{
    void* dP;
    aclrtMalloc(&dP, 60, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, (const float*)dP, nullptr, 5)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dP);
}

// L1 — error interception
TEST_F(StpttrTest, L1_33_handle_priority)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(nullptr, (aclblasFillMode_t)0xFF, -1, nullptr, nullptr, 3)),
        static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}
TEST_F(StpttrTest, L1_34_uplo_before_n0)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)0xFF, 0, nullptr, nullptr, 1)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
TEST_F(StpttrTest, L1_35_n0_overrides_lda)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 0, nullptr, nullptr, 0)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}
TEST_F(StpttrTest, L1_36_both_ptrs_null)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, 5, nullptr, nullptr, 5)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
TEST_F(StpttrTest, L1_37_uplo_120)
{
    void *d1, *d2;
    aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)120, 3, (const float*)d1, (float*)d2, 3)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(d1);
    aclrtFree(d2);
}
TEST_F(StpttrTest, L1_38_uplo_123)
{
    void *d1, *d2;
    aclrtMalloc(&d1, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStpttr(handle_, (aclblasFillMode_t)123, 3, (const float*)d1, (float*)d2, 3)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(d1);
    aclrtFree(d2);
}

// L1 — roundtrip
TEST_F(StpttrTest, L1_39_roundtrip_lower)
{
    uint32_t n = 32, lda = 32;
    size_t ap = n * (n + 1) / 2, al = lda * n;
    std::vector<float> s(al);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < n; i++)
            s[j * lda + i] = static_cast<float>(j * n + i + 1);
    void *dS, *dP, *dD;
    ASSERT_EQ(aclrtMalloc(&dS, al * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dP, ap * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, al * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dS, al * 4, s.data(), al * 4, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, n, (const float*)dS, lda, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    aclrtSynchronizeStream(stream_);
    ASSERT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_LOWER, n, (const float*)dP, (float*)dD, lda)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    aclrtSynchronizeStream(stream_);
    std::vector<float> r(al);
    ASSERT_EQ(aclrtMemcpy(r.data(), al * 4, dD, al * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = j; i < n; i++)
            ASSERT_EQ(r[j * lda + i], s[j * lda + i]);
    aclrtFree(dS);
    aclrtFree(dP);
    aclrtFree(dD);
}
TEST_F(StpttrTest, L1_40_roundtrip_upper)
{
    uint32_t n = 32, lda = 32;
    size_t ap = n * (n + 1) / 2, al = lda * n;
    std::vector<float> s(al);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < n; i++)
            s[j * lda + i] = static_cast<float>(j * n + i + 1);
    void *dS, *dP, *dD;
    ASSERT_EQ(aclrtMalloc(&dS, al * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dP, ap * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, al * 4, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dS, al * 4, s.data(), al * 4, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_UPPER, n, (const float*)dS, lda, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    aclrtSynchronizeStream(stream_);
    ASSERT_EQ(
        static_cast<int>(aclblasStpttr(handle_, ACLBLAS_UPPER, n, (const float*)dP, (float*)dD, lda)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    aclrtSynchronizeStream(stream_);
    std::vector<float> r(al);
    ASSERT_EQ(aclrtMemcpy(r.data(), al * 4, dD, al * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i <= j; i++)
            ASSERT_EQ(r[j * lda + i], s[j * lda + i]);
    aclrtFree(dS);
    aclrtFree(dP);
    aclrtFree(dD);
}

// L1 — extreme scale
TEST_F(StpttrTest, L1_41_n10240_lower) { EXPECT_TRUE(Run(10240, 10240, ACLBLAS_LOWER)); }
TEST_F(StpttrTest, L1_42_n10240_upper) { EXPECT_TRUE(Run(10240, 10240, ACLBLAS_UPPER)); }
