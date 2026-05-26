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

class StrttpTest : public ::testing::Test {
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

    static bool IsNanF(float x) { return x != x; }

    static void Golden(uint32_t n, uint32_t lda, uint32_t uplo, const std::vector<float>& a, std::vector<float>& ap)
    {
        size_t idx = 0;
        if (uplo == 0) {
            for (uint32_t j = 0; j < n; j++)
                for (uint32_t i = j; i < n; i++)
                    ap[idx++] = a[j * lda + i];
        } else {
            for (uint32_t j = 0; j < n; j++)
                for (uint32_t i = 0; i <= j; i++)
                    ap[idx++] = a[j * lda + i];
        }
    }

    void FillA(std::vector<float>& a, uint32_t n, uint32_t lda)
    {
        for (uint32_t j = 0; j < n; j++)
            for (uint32_t i = 0; i < n; i++)
                a[j * lda + i] = static_cast<float>(j * n + i + 1);
    }

    bool Run(uint32_t n, uint32_t lda, aclblasFillMode_t uplo)
    {
        std::vector<float> a(lda * n);
        FillA(a, n, lda);
        return Run(n, lda, uplo, a);
    }

    bool Run(uint32_t n, uint32_t lda, aclblasFillMode_t uplo, const std::vector<float>& a)
    {
        size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
        size_t aLen = static_cast<size_t>(lda) * n;
        std::vector<float> ap(apLen), gld(apLen);
        Golden(n, lda, uplo == ACLBLAS_LOWER ? 0u : 1u, a, gld);

        void *dA = nullptr, *dP = nullptr;
        aclError aclRet = aclrtMalloc(&dA, aLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS)
            return false;
        aclRet = aclrtMalloc(&dP, apLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dA);
            return false;
        }

        aclRet = aclrtMemcpy(dA, aLen * sizeof(float), a.data(), aLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dA);
            aclrtFree(dP);
            return false;
        }

        auto ret = aclblasStrttp(
            handle_, uplo, static_cast<int>(n), static_cast<const float*>(dA), static_cast<int>(lda),
            static_cast<float*>(dP));
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            aclrtFree(dA);
            aclrtFree(dP);
            return false;
        }
        aclrtSynchronizeStream(stream_);

        aclRet = aclrtMemcpy(ap.data(), apLen * sizeof(float), dP, apLen * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dA);
        aclrtFree(dP);
        if (aclRet != ACL_SUCCESS)
            return false;

        for (size_t i = 0; i < ap.size(); i++) {
            if (IsNanF(ap[i]) && IsNanF(gld[i]))
                continue;
            if (ap[i] != gld[i])
                return false;
        }
        return true;
    }
};

aclblasHandle_t StrttpTest::handle_ = nullptr;
aclrtStream StrttpTest::stream_ = nullptr;

// L0 — parameter validation
TEST_F(StrttpTest, L0_01_handle_null)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(nullptr, ACLBLAS_LOWER, 4, nullptr, 4, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}
TEST_F(StrttpTest, L0_02_n_negative)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, -1, nullptr, 1, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
TEST_F(StrttpTest, L0_03_lda_small)
{
    std::vector<float> a(8);
    FillA(a, 4, 2);
    void *dA, *dP;
    aclrtMalloc(&dA, 32, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dP, 40, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclrtMemcpy(dA, 32, a.data(), 32, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_UPPER, 4, (const float*)dA, 2, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
    aclrtFree(dP);
}
TEST_F(StrttpTest, L0_04_uplo_bad)
{
    std::vector<float> a(16);
    FillA(a, 4, 4);
    void *dA, *dP;
    aclrtMalloc(&dA, 64, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dP, 40, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclrtMemcpy(dA, 64, a.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, (aclblasFillMode_t)0xFF, 4, (const float*)dA, 4, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
    aclrtFree(dP);
}
TEST_F(StrttpTest, L0_05_n_zero)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 0, nullptr, 1, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

// L0 — normal shapes
TEST_F(StrttpTest, L0_06_n1_lower) { EXPECT_TRUE(Run(1, 1, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_07_n1_upper) { EXPECT_TRUE(Run(1, 1, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L0_08_n2_lower) { EXPECT_TRUE(Run(2, 2, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_09_n2_upper) { EXPECT_TRUE(Run(2, 2, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L0_10_n4_lower) { EXPECT_TRUE(Run(4, 4, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_11_n4_upper) { EXPECT_TRUE(Run(4, 4, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L0_12_n32_lower) { EXPECT_TRUE(Run(32, 32, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_13_n32_upper) { EXPECT_TRUE(Run(32, 32, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L0_14_n128_lower) { EXPECT_TRUE(Run(128, 128, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_15_n128_upper) { EXPECT_TRUE(Run(128, 128, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L0_16_n512_lower) { EXPECT_TRUE(Run(512, 512, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L0_17_n512_upper) { EXPECT_TRUE(Run(512, 512, ACLBLAS_UPPER)); }

// L1 — extended shapes
TEST_F(StrttpTest, L1_01_n8_lower) { EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_02_n8_upper) { EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_03_n9_lower) { EXPECT_TRUE(Run(9, 9, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_04_n9_upper) { EXPECT_TRUE(Run(9, 9, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_05_n64_lower) { EXPECT_TRUE(Run(64, 64, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_06_n64_upper) { EXPECT_TRUE(Run(64, 64, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_07_n100_lower) { EXPECT_TRUE(Run(100, 100, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_08_n100_upper) { EXPECT_TRUE(Run(100, 100, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_09_n256_lower) { EXPECT_TRUE(Run(256, 256, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_10_n256_upper) { EXPECT_TRUE(Run(256, 256, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_11_n500_lower) { EXPECT_TRUE(Run(500, 500, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_12_n500_upper) { EXPECT_TRUE(Run(500, 500, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_13_n1024_lower) { EXPECT_TRUE(Run(1024, 1024, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_14_n1024_upper) { EXPECT_TRUE(Run(1024, 1024, ACLBLAS_UPPER)); }

// L1 — non-compact lda
TEST_F(StrttpTest, L1_15_lda12_lower) { EXPECT_TRUE(Run(8, 12, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_16_lda12_upper) { EXPECT_TRUE(Run(8, 12, ACLBLAS_UPPER)); }
TEST_F(StrttpTest, L1_17_lda32_lower) { EXPECT_TRUE(Run(16, 32, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_18_lda32_upper) { EXPECT_TRUE(Run(16, 32, ACLBLAS_UPPER)); }

// L1 — special values
TEST_F(StrttpTest, L1_19_zeros_lower)
{
    std::vector<float> a(64, 0);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_20_zeros_upper)
{
    std::vector<float> a(64, 0);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}
TEST_F(StrttpTest, L1_21_large_lower)
{
    std::vector<float> a(64, 1e10f);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_22_large_upper)
{
    std::vector<float> a(64, 1e10f);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}
TEST_F(StrttpTest, L1_23_neg_lower)
{
    std::vector<float> a(64);
    for (size_t i = 0; i < 64; i++)
        a[i] = -static_cast<float>(i + 1);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_24_neg_upper)
{
    std::vector<float> a(64);
    for (size_t i = 0; i < 64; i++)
        a[i] = -static_cast<float>(i + 1);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}
TEST_F(StrttpTest, L1_25_inf_lower)
{
    std::vector<float> a(64, INFINITY);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_26_inf_upper)
{
    std::vector<float> a(64, INFINITY);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}
TEST_F(StrttpTest, L1_27_nan_lower)
{
    std::vector<float> a(64, NAN);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_28_nan_upper)
{
    std::vector<float> a(64, NAN);
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}
TEST_F(StrttpTest, L1_29_extr_lower)
{
    std::vector<float> a(64);
    float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (size_t i = 0; i < 64; i++)
        a[i] = v[i % 7];
    EXPECT_TRUE(Run(8, 8, ACLBLAS_LOWER, a));
}
TEST_F(StrttpTest, L1_30_extr_upper)
{
    std::vector<float> a(64);
    float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (size_t i = 0; i < 64; i++)
        a[i] = v[i % 7];
    EXPECT_TRUE(Run(8, 8, ACLBLAS_UPPER, a));
}

// L1 — null pointers
TEST_F(StrttpTest, L1_31_a_null)
{
    void* dP;
    aclrtMalloc(&dP, 40, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 4, nullptr, 4, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dP);
}
TEST_F(StrttpTest, L1_32_ap_null)
{
    std::vector<float> a(16);
    FillA(a, 4, 4);
    void* dA;
    aclrtMalloc(&dA, 64, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclrtMemcpy(dA, 64, a.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 4, (const float*)dA, 4, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
}

// L1 — error interception
TEST_F(StrttpTest, L1_33_handle_priority)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(nullptr, (aclblasFillMode_t)0xFF, -1, nullptr, 3, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}
TEST_F(StrttpTest, L1_34_uplo_before_n0)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, (aclblasFillMode_t)0xFF, 0, nullptr, 1, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
TEST_F(StrttpTest, L1_35_n0_overrides_lda)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 0, nullptr, 0, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}
TEST_F(StrttpTest, L1_36_both_ptrs_null)
{
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, ACLBLAS_LOWER, 5, nullptr, 5, nullptr)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
TEST_F(StrttpTest, L1_37_uplo_120)
{
    std::vector<float> a(9);
    FillA(a, 3, 3);
    void *dA, *dP;
    aclrtMalloc(&dA, 36, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dP, 24, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclrtMemcpy(dA, 36, a.data(), 36, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, (aclblasFillMode_t)120, 3, (const float*)dA, 3, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
    aclrtFree(dP);
}
TEST_F(StrttpTest, L1_38_uplo_123)
{
    std::vector<float> a(9);
    FillA(a, 3, 3);
    void *dA, *dP;
    aclrtMalloc(&dA, 36, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dP, 24, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(aclrtMemcpy(dA, 36, a.data(), 36, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(
        static_cast<int>(aclblasStrttp(handle_, (aclblasFillMode_t)123, 3, (const float*)dA, 3, (float*)dP)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
    aclrtFree(dP);
}

// L1 — roundtrip
TEST_F(StrttpTest, L1_39_roundtrip_lower)
{
    uint32_t n = 32, lda = 32;
    size_t ap = n * (n + 1) / 2, al = lda * n;
    std::vector<float> s(al);
    FillA(s, n, lda);
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
TEST_F(StrttpTest, L1_40_roundtrip_upper)
{
    uint32_t n = 32, lda = 32;
    size_t ap = n * (n + 1) / 2, al = lda * n;
    std::vector<float> s(al);
    FillA(s, n, lda);
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
TEST_F(StrttpTest, L1_41_n10240_lower) { EXPECT_TRUE(Run(10240, 10240, ACLBLAS_LOWER)); }
TEST_F(StrttpTest, L1_42_n10240_upper) { EXPECT_TRUE(Run(10240, 10240, ACLBLAS_UPPER)); }
