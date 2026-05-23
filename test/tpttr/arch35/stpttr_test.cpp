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
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include "acl/acl.h"
#include "cann_ops_blas.h"

constexpr float kSentinel = -999.0f;

// ============================================================================
// RAII helpers
// ============================================================================
struct DeviceMem
{
    void* ptr = nullptr;
    explicit DeviceMem(size_t bytes) { aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST); }
    ~DeviceMem() { if (ptr) aclrtFree(ptr); }
    bool ok() const { return ptr != nullptr; }
    template<typename T> T* as() { return reinterpret_cast<T*>(ptr); }
};

struct TestCtx
{
    aclblasHandle_t handle = nullptr;
    aclrtStream    stream = nullptr;
    TestCtx()
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclblasCreate(&handle);
        aclrtCreateStream(&stream);
        aclblasSetStream(handle, stream);
    }
    ~TestCtx()
    {
        aclrtDestroyStream(stream);
        aclblasDestroy(handle);
        aclrtResetDevice(0);
        aclFinalize();
    }
};

struct TestLog
{
    int pass = 0;
    int fail = 0;

    void ok(bool cond, const char* name)
    {
        if (cond) {
            pass++;
            std::cout << "[PASS] " << name << std::endl;
        } else {
            fail++;
            std::cout << "[FAIL] " << name << std::endl;
        }
    }

    void expectStatus(int expected, int actual, const char* name)
    {
        ok(actual == expected, name);
        if (actual != expected) {
            std::cout << "       expected " << expected << " got " << actual << std::endl;
        }
    }

    int done()
    {
        std::cout << "\n" << pass << "/" << (pass + fail) << " passed";
        if (fail) {
            std::cout << " (" << fail << " failed)";
        }
        std::cout << std::endl;
        return fail ? 1 : 0;
    }
};

// ============================================================================
// Golden reference: packed AP → full A
// ============================================================================
static void Golden(int n, aclblasFillMode_t uplo, const float* ap, int lda, float* a)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < lda; i++) {
            a[j * lda + i] = kSentinel;
        }
    }
    int idx = 0;
    if (uplo == ACLBLAS_LOWER) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                a[j * lda + i] = ap[idx++];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                a[j * lda + i] = ap[idx++];
            }
        }
    }
}

static bool IsNanF(float x) { return x != x; }

static bool Verify(const std::vector<float>& out, const std::vector<float>& gold)
{
    if (out.size() != gold.size()) {
        std::cout << "  size mismatch" << std::endl;
        return false;
    }
    for (size_t i = 0; i < out.size(); i++) {
        if (IsNanF(out[i]) && IsNanF(gold[i])) {
            continue;
        }
        if (out[i] != gold[i]) {
            std::cout << "  mismatch at " << i << ": " << out[i] << " vs " << gold[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Test runner: alloc device memory, call kernel, verify
// ============================================================================
static bool RunCase(aclblasHandle_t h, aclrtStream s, int n, int lda, aclblasFillMode_t uplo,
                    const std::vector<float>& ap)
{
    int aLen = lda * n;
    std::vector<float> a(aLen, kSentinel);
    std::vector<float> golden(aLen);
    Golden(n, uplo, ap.data(), lda, golden.data());

    DeviceMem dAP(ap.size() * sizeof(float));
    DeviceMem dA(aLen * sizeof(float));
    if (!dAP.ok() || !dA.ok()) {
        std::cout << "  malloc failed" << std::endl;
        return false;
    }
    aclrtMemcpy(dAP.ptr, ap.size() * sizeof(float), ap.data(), ap.size() * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dA.ptr, aLen * sizeof(float), a.data(), aLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    auto ret = aclblastpttr(h, uplo, n, dAP.as<const float>(), dA.as<float>(), lda);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "  kernel returned " << ret << std::endl;
        return false;
    }
    aclrtSynchronizeStream(s);

    std::vector<float> out(aLen);
    aclrtMemcpy(out.data(), aLen * sizeof(float), dA.ptr, aLen * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    return Verify(out, golden);
}

static bool RunCase(aclblasHandle_t h, aclrtStream s, int n, int lda, aclblasFillMode_t uplo)
{
    std::vector<float> ap(n * (n + 1) / 2);
    for (size_t i = 0; i < ap.size(); i++) {
        ap[i] = static_cast<float>(i + 1);
    }
    return RunCase(h, s, n, lda, uplo, ap);
}

// ============================================================================
// L0-01  handle==nullptr → NOT_INITIALIZED
// ============================================================================
static void t_L0_01_handle_null(TestLog& log)
{
    auto r = aclblastpttr(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    log.expectStatus(ACLBLAS_STATUS_NOT_INITIALIZED, r,
                     "L0-01 handle=nullptr -> NOT_INITIALIZED");
}

// ============================================================================
// L0-02  n < 0 → INVALID_VALUE
// ============================================================================
static void t_L0_02_n_negative(TestCtx& ctx, TestLog& log)
{
    DeviceMem d1(sizeof(float));
    DeviceMem d2(sizeof(float));
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, -1, d1.as<const float>(), d2.as<float>(), 1);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L0-02 n<0 -> INVALID_VALUE");
}

// ============================================================================
// L0-03  lda < max(1, n) → INVALID_VALUE   (n=5, lda=3)
// ============================================================================
static void t_L0_03_lda_small(TestCtx& ctx, TestLog& log)
{
    DeviceMem dAP(15 * sizeof(float));
    DeviceMem dA(15 * sizeof(float));
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, 5, dAP.as<const float>(), dA.as<float>(), 3);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L0-03 lda<max(1,n) -> INVALID_VALUE");
}

// ============================================================================
// L0-04  invalid uplo → INVALID_VALUE
// ============================================================================
static void t_L0_04_uplo_bad(TestCtx& ctx, TestLog& log)
{
    DeviceMem d1(sizeof(float));
    DeviceMem d2(sizeof(float));
    auto r = aclblastpttr(ctx.handle, static_cast<aclblasFillMode_t>(0xFF), 5,
                          d1.as<const float>(), d2.as<float>(), 5);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L0-04 uplo invalid -> INVALID_VALUE");
}

// ============================================================================
// L0-05  n == 0 → SUCCESS
// ============================================================================
static void t_L0_05_n0(TestCtx& ctx, TestLog& log)
{
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, 0, nullptr, nullptr, 1);
    log.expectStatus(ACLBLAS_STATUS_SUCCESS, r, "L0-05 n=0 -> SUCCESS");
}

// ============================================================================
// L0 — normal shapes (12)
// ============================================================================
static void t_L0_06_n1_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 1, 1, ACLBLAS_LOWER), "L0-06 n=1 LOWER");
}
static void t_L0_07_n1_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 1, 1, ACLBLAS_UPPER), "L0-07 n=1 UPPER");
}
static void t_L0_08_n2_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 2, 2, ACLBLAS_LOWER), "L0-08 n=2 LOWER");
}
static void t_L0_09_n2_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 2, 2, ACLBLAS_UPPER), "L0-09 n=2 UPPER");
}
static void t_L0_10_n4_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 4, 4, ACLBLAS_LOWER), "L0-10 n=4 LOWER");
}
static void t_L0_11_n4_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 4, 4, ACLBLAS_UPPER), "L0-11 n=4 UPPER");
}
static void t_L0_12_n32_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 32, 32, ACLBLAS_LOWER), "L0-12 n=32 LOWER");
}
static void t_L0_13_n32_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 32, 32, ACLBLAS_UPPER), "L0-13 n=32 UPPER");
}
static void t_L0_14_n128_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 128, 128, ACLBLAS_LOWER), "L0-14 n=128 LOWER");
}
static void t_L0_15_n128_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 128, 128, ACLBLAS_UPPER), "L0-15 n=128 UPPER");
}
static void t_L0_16_n512_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 512, 512, ACLBLAS_LOWER), "L0-16 n=512 LOWER");
}
static void t_L0_17_n512_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 512, 512, ACLBLAS_UPPER), "L0-17 n=512 UPPER");
}

// ============================================================================
// L1 — extended shapes (14)
// ============================================================================
static void t_L1_01_n8_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 8, 8, ACLBLAS_LOWER), "L1-01 n=8 LOWER");
}
static void t_L1_02_n8_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 8, 8, ACLBLAS_UPPER), "L1-02 n=8 UPPER");
}
static void t_L1_03_n9_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 9, 9, ACLBLAS_LOWER), "L1-03 n=9 LOWER (odd)");
}
static void t_L1_04_n9_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 9, 9, ACLBLAS_UPPER), "L1-04 n=9 UPPER (odd)");
}
static void t_L1_05_n64_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 64, 64, ACLBLAS_LOWER), "L1-05 n=64 LOWER");
}
static void t_L1_06_n64_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 64, 64, ACLBLAS_UPPER), "L1-06 n=64 UPPER");
}
static void t_L1_07_n100_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 100, 100, ACLBLAS_LOWER), "L1-07 n=100 LOWER");
}
static void t_L1_08_n100_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 100, 100, ACLBLAS_UPPER), "L1-08 n=100 UPPER");
}
static void t_L1_09_n256_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 256, 256, ACLBLAS_LOWER), "L1-09 n=256 LOWER");
}
static void t_L1_10_n256_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 256, 256, ACLBLAS_UPPER), "L1-10 n=256 UPPER");
}
static void t_L1_11_n500_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 500, 500, ACLBLAS_LOWER), "L1-11 n=500 LOWER");
}
static void t_L1_12_n500_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 500, 500, ACLBLAS_UPPER), "L1-12 n=500 UPPER");
}
static void t_L1_13_n1024_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 1024, 1024, ACLBLAS_LOWER), "L1-13 n=1024 LOWER");
}
static void t_L1_14_n1024_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 1024, 1024, ACLBLAS_UPPER), "L1-14 n=1024 UPPER");
}

// ============================================================================
// L1 — non-compact lda (4)
// ============================================================================
static void t_L1_15_lda_large_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 8, 12, ACLBLAS_LOWER), "L1-15 lda=12 n=8 LOWER");
}
static void t_L1_16_lda_large_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 8, 12, ACLBLAS_UPPER), "L1-16 lda=12 n=8 UPPER");
}
static void t_L1_17_lda_xlarge_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 16, 32, ACLBLAS_LOWER), "L1-17 lda=32 n=16 LOWER");
}
static void t_L1_18_lda_xlarge_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 16, 32, ACLBLAS_UPPER), "L1-18 lda=32 n=16 UPPER");
}

// ============================================================================
// L1 — special values (12)
// ============================================================================
static void t_L1_19_zeros_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, 0.0f);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-19 zeros LOWER");
}
static void t_L1_20_zeros_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, 0.0f);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-20 zeros UPPER");
}
static void t_L1_21_large_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, 1.0e10f);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-21 large pos LOWER");
}
static void t_L1_22_large_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, 1.0e10f);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-22 large pos UPPER");
}
static void t_L1_23_neg_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    int len = n * (n + 1) / 2;
    std::vector<float> ap(len);
    for (int i = 0; i < len; i++) {
        ap[i] = -static_cast<float>(i + 1);
    }
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-23 negative LOWER");
}
static void t_L1_24_neg_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    int len = n * (n + 1) / 2;
    std::vector<float> ap(len);
    for (int i = 0; i < len; i++) {
        ap[i] = -static_cast<float>(i + 1);
    }
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-24 negative UPPER");
}
static void t_L1_25_inf_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, INFINITY);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-25 INF LOWER");
}
static void t_L1_26_inf_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, INFINITY);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-26 INF UPPER");
}
static void t_L1_27_nan_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, NAN);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-27 NAN LOWER");
}
static void t_L1_28_nan_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    std::vector<float> ap(n * (n + 1) / 2, NAN);
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-28 NAN UPPER");
}
static void t_L1_29_extreme_lower(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    int len = n * (n + 1) / 2;
    std::vector<float> ap(len);
    float vals[] = {1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (int i = 0; i < len; i++) {
        ap[i] = vals[i % 7];
    }
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_LOWER, ap), "L1-29 extreme LOWER");
}
static void t_L1_30_extreme_upper(TestCtx& ctx, TestLog& log)
{
    int n = 8;
    int len = n * (n + 1) / 2;
    std::vector<float> ap(len);
    float vals[] = {1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
    for (int i = 0; i < len; i++) {
        ap[i] = vals[i % 7];
    }
    log.ok(RunCase(ctx.handle, ctx.stream, n, n, ACLBLAS_UPPER, ap), "L1-30 extreme UPPER");
}

// ============================================================================
// L1 — null pointers (2)
// ============================================================================
static void t_L1_31_ap_null(TestCtx& ctx, TestLog& log)
{
    int n = 5;
    int lda = 5;
    DeviceMem dA(lda * n * sizeof(float));
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, n, nullptr, dA.as<float>(), lda);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L1-31 AP=nullptr -> INVALID_VALUE");
}
static void t_L1_32_a_null(TestCtx& ctx, TestLog& log)
{
    int n = 5;
    int lda = 5;
    DeviceMem dAP(n * (n + 1) / 2 * sizeof(float));
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, n, dAP.as<const float>(), nullptr, lda);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L1-32 A=nullptr -> INVALID_VALUE");
}

// ============================================================================
// L1 — error interception: priority & boundary (6)
// ============================================================================
static void t_L1_33_handle_priority(TestLog& log)
{
    // handle=nullptr overrides all other errors (n<0, uplo bad, lda bad, ptrs null)
    auto r = aclblastpttr(nullptr, static_cast<aclblasFillMode_t>(0xFF), -1, nullptr, nullptr, 3);
    log.expectStatus(ACLBLAS_STATUS_NOT_INITIALIZED, r,
                     "L1-33 handle=null overrides all other errors");
}
static void t_L1_34_uplo_before_n0(TestCtx& ctx, TestLog& log)
{
    // uplo checked before n==0 early return → INVALID_VALUE takes priority
    auto r = aclblastpttr(ctx.handle, static_cast<aclblasFillMode_t>(0xFF), 0, nullptr, nullptr, 1);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r,
                     "L1-34 uplo invalid checked before n=0");
}
static void t_L1_35_n0_overrides_lda(TestCtx& ctx, TestLog& log)
{
    // n=0 checked before lda → SUCCESS even with lda < max(1,n)
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, 0, nullptr, nullptr, 0);
    log.expectStatus(ACLBLAS_STATUS_SUCCESS, r, "L1-35 n=0 overrides lda<max(1,n)");
}
static void t_L1_36_both_ptrs_null(TestCtx& ctx, TestLog& log)
{
    // Both AP and A are null with n>0 → INVALID_VALUE
    auto r = aclblastpttr(ctx.handle, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L1-36 both ptrs null");
}
static void t_L1_37_uplo_boundary_lt(TestCtx& ctx, TestLog& log)
{
    // Boundary just below valid range: uplo=120 (UPPER=121, LOWER=122)
    DeviceMem d1(sizeof(float));
    DeviceMem d2(sizeof(float));
    auto r = aclblastpttr(ctx.handle, static_cast<aclblasFillMode_t>(120), 3,
                          d1.as<const float>(), d2.as<float>(), 3);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L1-37 uplo=120 (below valid range)");
}
static void t_L1_38_uplo_boundary_gt(TestCtx& ctx, TestLog& log)
{
    // Boundary just above valid range: uplo=123 (UPPER=121, LOWER=122)
    DeviceMem d1(sizeof(float));
    DeviceMem d2(sizeof(float));
    auto r = aclblastpttr(ctx.handle, static_cast<aclblasFillMode_t>(123), 3,
                          d1.as<const float>(), d2.as<float>(), 3);
    log.expectStatus(ACLBLAS_STATUS_INVALID_VALUE, r, "L1-38 uplo=123 (above valid range)");
}

// ============================================================================
// L1 — roundtrip: trttp(A)→AP, tpttr(AP)→A', verify A'==A (2)
// ============================================================================
static void t_L1_39_roundtrip_lower(TestCtx& ctx, TestLog& log)
{
    uint32_t n = 32;
    uint32_t lda = 32;
    size_t apLen = n * (n + 1) / 2;
    size_t aLen = lda * n;
    std::vector<float> aSrc(aLen);
    std::vector<float> ap(apLen);
    std::vector<float> aDst(aLen);
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            aSrc[j * lda + i] = static_cast<float>(j * n + i + 1);
        }
    }

    DeviceMem dASrc(aLen * sizeof(float));
    DeviceMem dAP(apLen * sizeof(float));
    DeviceMem dADst(aLen * sizeof(float));
    if (!dASrc.ok() || !dAP.ok() || !dADst.ok()) {
        log.ok(false, "L1-39 roundtrip LOWER");
        return;
    }
    aclrtMemcpy(dASrc.ptr, aLen * sizeof(float), aSrc.data(), aLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    auto r1 = aclblasStrttp(ctx.handle, ACLBLAS_LOWER, n, dASrc.as<const float>(), lda,
                            dAP.as<float>());
    if (r1 != ACLBLAS_STATUS_SUCCESS) {
        log.ok(false, "L1-39 roundtrip (trttp)");
        return;
    }
    aclrtSynchronizeStream(ctx.stream);

    auto r2 = aclblastpttr(ctx.handle, ACLBLAS_LOWER, n, dAP.as<const float>(),
                           dADst.as<float>(), lda);
    if (r2 != ACLBLAS_STATUS_SUCCESS) {
        log.ok(false, "L1-39 roundtrip (tpttr)");
        return;
    }
    aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(aDst.data(), aLen * sizeof(float), dADst.ptr, aLen * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = true;
    for (uint32_t j = 0; j < n && ok; j++) {
        for (uint32_t i = j; i < n; i++) {
            if (aDst[j * lda + i] != aSrc[j * lda + i]) {
                ok = false;
            }
        }
    }
    log.ok(ok, "L1-39 roundtrip LOWER");
}
static void t_L1_40_roundtrip_upper(TestCtx& ctx, TestLog& log)
{
    uint32_t n = 32;
    uint32_t lda = 32;
    size_t apLen = n * (n + 1) / 2;
    size_t aLen = lda * n;
    std::vector<float> aSrc(aLen);
    std::vector<float> ap(apLen);
    std::vector<float> aDst(aLen);
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            aSrc[j * lda + i] = static_cast<float>(j * n + i + 1);
        }
    }

    DeviceMem dASrc(aLen * sizeof(float));
    DeviceMem dAP(apLen * sizeof(float));
    DeviceMem dADst(aLen * sizeof(float));
    if (!dASrc.ok() || !dAP.ok() || !dADst.ok()) {
        log.ok(false, "L1-40 roundtrip UPPER");
        return;
    }
    aclrtMemcpy(dASrc.ptr, aLen * sizeof(float), aSrc.data(), aLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    auto r1 = aclblasStrttp(ctx.handle, ACLBLAS_UPPER, n, dASrc.as<const float>(), lda,
                            dAP.as<float>());
    if (r1 != ACLBLAS_STATUS_SUCCESS) {
        log.ok(false, "L1-40 roundtrip (trttp)");
        return;
    }
    aclrtSynchronizeStream(ctx.stream);

    auto r2 = aclblastpttr(ctx.handle, ACLBLAS_UPPER, n, dAP.as<const float>(),
                           dADst.as<float>(), lda);
    if (r2 != ACLBLAS_STATUS_SUCCESS) {
        log.ok(false, "L1-40 roundtrip (tpttr)");
        return;
    }
    aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(aDst.data(), aLen * sizeof(float), dADst.ptr, aLen * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = true;
    for (uint32_t j = 0; j < n && ok; j++) {
        for (uint32_t i = 0; i <= j && ok; i++) {
            if (aDst[j * lda + i] != aSrc[j * lda + i]) {
                ok = false;
            }
        }
    }
    log.ok(ok, "L1-40 roundtrip UPPER");
}

// ============================================================================
// L1 — extreme scale (2)
// ============================================================================
static void t_L1_41_n10240_lower(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 10240, 10240, ACLBLAS_LOWER), "L1-41 n=10240 LOWER");
}
static void t_L1_42_n10240_upper(TestCtx& ctx, TestLog& log)
{
    log.ok(RunCase(ctx.handle, ctx.stream, 10240, 10240, ACLBLAS_UPPER), "L1-42 n=10240 UPPER");
}

static void runL0Tests(TestCtx& ctx, TestLog& log)
{
    std::cout << "=== aclblastpttr L0 ===" << std::endl;
    t_L0_02_n_negative(ctx, log);
    t_L0_03_lda_small(ctx, log);
    t_L0_04_uplo_bad(ctx, log);
    t_L0_05_n0(ctx, log);
    t_L0_06_n1_lower(ctx, log);
    t_L0_07_n1_upper(ctx, log);
    t_L0_08_n2_lower(ctx, log);
    t_L0_09_n2_upper(ctx, log);
    t_L0_10_n4_lower(ctx, log);
    t_L0_11_n4_upper(ctx, log);
    t_L0_12_n32_lower(ctx, log);
    t_L0_13_n32_upper(ctx, log);
    t_L0_14_n128_lower(ctx, log);
    t_L0_15_n128_upper(ctx, log);
    t_L0_16_n512_lower(ctx, log);
    t_L0_17_n512_upper(ctx, log);
}

static void runL1Tests(TestCtx& ctx, TestLog& log)
{
    std::cout << "\n=== aclblastpttr L1 ===" << std::endl;
    t_L1_01_n8_lower(ctx, log);
    t_L1_02_n8_upper(ctx, log);
    t_L1_03_n9_lower(ctx, log);
    t_L1_04_n9_upper(ctx, log);
    t_L1_05_n64_lower(ctx, log);
    t_L1_06_n64_upper(ctx, log);
    t_L1_07_n100_lower(ctx, log);
    t_L1_08_n100_upper(ctx, log);
    t_L1_09_n256_lower(ctx, log);
    t_L1_10_n256_upper(ctx, log);
    t_L1_11_n500_lower(ctx, log);
    t_L1_12_n500_upper(ctx, log);
    t_L1_13_n1024_lower(ctx, log);
    t_L1_14_n1024_upper(ctx, log);
    t_L1_15_lda_large_lower(ctx, log);
    t_L1_16_lda_large_upper(ctx, log);
    t_L1_17_lda_xlarge_lower(ctx, log);
    t_L1_18_lda_xlarge_upper(ctx, log);
    t_L1_19_zeros_lower(ctx, log);
    t_L1_20_zeros_upper(ctx, log);
    t_L1_21_large_lower(ctx, log);
    t_L1_22_large_upper(ctx, log);
    t_L1_23_neg_lower(ctx, log);
    t_L1_24_neg_upper(ctx, log);
    t_L1_25_inf_lower(ctx, log);
    t_L1_26_inf_upper(ctx, log);
    t_L1_27_nan_lower(ctx, log);
    t_L1_28_nan_upper(ctx, log);
    t_L1_29_extreme_lower(ctx, log);
    t_L1_30_extreme_upper(ctx, log);
    t_L1_31_ap_null(ctx, log);
    t_L1_32_a_null(ctx, log);
    t_L1_33_handle_priority(log);
    t_L1_34_uplo_before_n0(ctx, log);
    t_L1_35_n0_overrides_lda(ctx, log);
    t_L1_36_both_ptrs_null(ctx, log);
    t_L1_37_uplo_boundary_lt(ctx, log);
    t_L1_38_uplo_boundary_gt(ctx, log);
    t_L1_39_roundtrip_lower(ctx, log);
    t_L1_40_roundtrip_upper(ctx, log);
    t_L1_41_n10240_lower(ctx, log);
    t_L1_42_n10240_upper(ctx, log);
}

// ============================================================================
int main()
{
    TestLog log;
    t_L0_01_handle_null(log);
    TestCtx ctx;
    runL0Tests(ctx, log);
    runL1Tests(ctx, log);
    return log.done();
}
