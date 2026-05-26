/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

constexpr float MERE_THRESHOLD = 1.220703125e-4f;
constexpr float MARE_RATIO = 10.0f;

// Golden: y = alpha * A * x + beta * y
// A is column-major, symmetric, uplo indicates which triangle is stored.
static void GoldenSymv(bool isUpper, const std::vector<float>& A, int64_t n, int64_t lda, float alpha,
                       const std::vector<float>& x, int64_t incx,
                       float beta, std::vector<float>& y, int64_t incy)
{
    std::vector<float> yOrig(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t yIdx = (incy >= 0) ? (i * incy) : ((n - 1 - i) * (-incy));
        yOrig[i] = y[yIdx];
    }

    for (int64_t row = 0; row < n; ++row) {
        float acc = 0.0f;
        for (int64_t col = 0; col < n; ++col) {
            bool stored = isUpper ? (row <= col) : (row >= col);
            float aVal = stored ? A[row + col * lda] : A[col + row * lda];
            float xVal = (incx >= 0) ? x[col * incx] : x[(n - 1 - col) * (-incx)];
            acc += aVal * xVal;
        }
        int64_t yIdx = (incy >= 0) ? (row * incy) : ((n - 1 - row) * (-incy));
        y[yIdx] = alpha * acc + beta * yOrig[row];
    }
}

static uint32_t VerifyResult(const float* out, const float* gold, int64_t n, int64_t incy, const char* nm)
{
    if (n == 0) {
        LOG_PRINT("  [%s] PASSED (Empty Result)\n", nm);
        return 0;
    }
    double sumRelErr = 0.0;
    float maxRelErr = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        int64_t idx = (incy >= 0) ? (i * incy) : ((n - 1 - i) * (-incy));
        float relErr = std::abs(out[idx] - gold[idx]) / (std::abs(gold[idx]) + 1e-7f);
        sumRelErr += relErr;
        if (relErr > maxRelErr)
            maxRelErr = relErr;
    }
    float mere = static_cast<float>(sumRelErr / n);
    float mare = maxRelErr;
    float mareThreshold = MARE_RATIO * MERE_THRESHOLD;
    bool pass = (mere < MERE_THRESHOLD) && (mare < mareThreshold);
    if (pass)
        LOG_PRINT("  [%s] PASSED (MERE=%.6e, MARE=%.6e)\n", nm, mere, mare);
    else
        LOG_PRINT(
            "  [%s] FAILED (MERE=%.6e, MARE=%.6e, limit=%.6e/%.6e)\n", nm, mere, mare, MERE_THRESHOLD, mareThreshold);
    return pass ? 0 : 1;
}

static aclblasStatus_t Run(aclblasHandle h, aclrtStream s,
                            aclblasFillMode uplo, int n,
                            float alpha, const std::vector<float>& A, int lda,
                            const std::vector<float>& x, int incx,
                            float beta, std::vector<float>& y, int incy)
{
    size_t aBytes = static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(float);
    size_t xBytes = (1 + (static_cast<size_t>(n) - 1) * static_cast<size_t>(std::abs(incx))) * sizeof(float);
    size_t yBytes = (1 + (static_cast<size_t>(n) - 1) * static_cast<size_t>(std::abs(incy))) * sizeof(float);

    float *dA = nullptr, *dx = nullptr, *dy = nullptr;
    if (aclrtMalloc((void**)&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    if (aclrtMalloc((void**)&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { aclrtFree(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc((void**)&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { aclrtFree(dA); aclrtFree(dx); return ACLBLAS_STATUS_ALLOC_FAILED; }

    aclrtMemcpy(dA, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dx, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dy, yBytes, y.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    auto ret = aclblasSsymv(h, uplo, n, &alpha, dA, lda, dx, incx, &beta, dy, incy);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtSynchronizeStream(s);
        aclrtMemcpy(y.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclrtFree(dA); aclrtFree(dx); aclrtFree(dy);
    return ret;
}

// === L0: Basic Functionality ===

static uint32_t L001(aclblasHandle h, aclrtStream s) {
    int n = 128, lda = n, incx = 1, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(n), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = i + 1;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 100) - 50.0f;
    std::vector<float> yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-01");
}

static uint32_t L002(aclblasHandle h, aclrtStream s) {
    int n = 128, lda = n, incx = 1, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(n), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = i + 1;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 100) - 50.0f;
    std::vector<float> yOut = y, yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-02");
}

static uint32_t L003(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = 1, incy = 1; float alpha = 0.0f, beta = 1.0f;
    std::vector<float> A(n * lda), x(n), y(n);
    for (int64_t i = 0; i < n; ++i) { x[i] = i + 1; y[i] = i + 1; }
    for (int64_t i = 0; i < n * lda; ++i) A[i] = i + 1;
    auto yGold = y;
    auto yOut = y;
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-03");
}

static uint32_t L004(aclblasHandle h, aclrtStream s) {
    int n = 1, lda = 1, incx = 1, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A = {3.0f}, x = {5.0f}, y = {0.0f};
    auto yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-04");
}

static uint32_t L005(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = 1, incy = 1; float alpha = -1.5f, beta = 0.5f;
    std::vector<float> A(n * lda), x(n), y(n);
    for (int64_t i = 0; i < n; ++i) { x[i] = (i % 10) * 1.0f; y[i] = (i % 7) * 1.0f; }
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 17) - 8.0f;
    std::vector<float> yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    uint32_t e = VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-05-UPPER");
    yOut = y; yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return e + VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L0-05-LOWER");
}

static uint32_t L006(aclblasHandle h, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, 0, &alpha, &dummy, 1, &dummy, 1, &alpha, &dummy, 1);
    if (ret == ACLBLAS_STATUS_SUCCESS) { LOG_PRINT("[SYMV-L0-06] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L0-06] FAILED\n"); return 1;
}

// === L1: Parameter Combinations ===

static uint32_t L101(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = 2, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(1 + (n - 1) * incx), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i * incx] = i + 1;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 50) - 25.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-01");
}

static uint32_t L102(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = -1, absIncx = 1, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(1 + (n - 1) * absIncx), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i * absIncx] = (n - 1 - i) + 1;  // reversed: x[0]=n, x[n-1]=1
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 100) - 50.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-02");
}

static uint32_t L103(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = -2, absIncx = 2, incy = 1; float alpha = 1.5f, beta = 0.0f;
    std::vector<float> A(n * lda), x(1 + (n - 1) * absIncx), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i * absIncx] = ((n - 1 - i) * 3) % 11 * 1.0f;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 13) * 1.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-03");
}

static uint32_t L104(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = 1, incy = 2; float alpha = 1.0f, beta = 1.5f;
    std::vector<float> A(n * lda), x(n), y(1 + (n - 1) * incy);
    for (int64_t i = 0; i < n; ++i) { x[i] = (i * 3) % 11 * 1.0f; y[i * incy] = (i % 5) * 1.0f; }
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 20) - 10.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-04");
}

static uint32_t L105(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n, incx = 1, incy = -1, absIncy = 1; float alpha = 1.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(n), y(1 + (n - 1) * absIncy, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = i + 1;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 30) - 15.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-05");
}

static uint32_t L106(aclblasHandle h, aclrtStream s) {
    int n = 64, lda = n + 32, incx = 1, incy = 1; float alpha = 2.0f, beta = 0.0f;
    std::vector<float> A(n * lda), x(n), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = i + 1;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 100) - 50.0f;
    auto yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-06");
}

static uint32_t L107(aclblasHandle h, aclrtStream s) {
    int n = 512, lda = n, incx = 1, incy = 1; float alpha = 0.5f, beta = 0.0f;
    std::vector<float> A(n * lda), x(n), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = (i % 31) * 1.0f;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 41) * 1.0f;
    std::vector<float> yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    uint32_t e = VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-07-UPPER");
    yOut = y; yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return e + VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-07-LOWER");
}

static uint32_t L108(aclblasHandle h, aclrtStream s) {
    int n = 4096, lda = n, incx = 1, incy = 1; float alpha = 1.0f, beta = 0.0f;
    LOG_PRINT("[SYMV-L1-08] Allocating A: %lld MB\n", (long long)(n * lda * 4 / 1024 / 1024));
    std::vector<float> A(n * lda), x(n), y(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) x[i] = (i % 997) * 1.0f;
    for (int64_t i = 0; i < n * lda; ++i) A[i] = (i % 503) * 1.0f;
    LOG_PRINT("[SYMV-L1-08-UPPER] Computing golden...\n");
    std::vector<float> yOut = y, yGold = y;
    GoldenSymv(true,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_UPPER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    uint32_t e = VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-08-UPPER");
    LOG_PRINT("[SYMV-L1-08-LOWER] Computing golden...\n");
    yOut = y; yGold = y;
    GoldenSymv(false,A, n, lda, alpha, x, incx, beta, yGold, incy);
    Run(h, s, ACLBLAS_LOWER, n, alpha, A, lda, x, incx, beta, yOut, incy);
    return e + VerifyResult(yOut.data(), yGold.data(), n, incy, "SYMV-L1-08-LOWER");
}

// === L2: Error Cases ===

static uint32_t L201(aclblasHandle h, aclrtStream) {
    float dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, -1, &dummy, &dummy, 1, &dummy, 1, &dummy, &dummy, 1);
    if (ret == ACLBLAS_STATUS_INVALID_VALUE) { LOG_PRINT("[SYMV-L2-01] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-01] FAILED\n"); return 1;
}

static uint32_t L202(aclblasHandle h, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, 4, &alpha, &dummy, 4, &dummy, 0, &alpha, &dummy, 4);
    if (ret == ACLBLAS_STATUS_INVALID_VALUE) { LOG_PRINT("[SYMV-L2-02] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-02] FAILED\n"); return 1;
}

static uint32_t L203(aclblasHandle h, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, 4, &alpha, &dummy, 4, &dummy, 1, &alpha, &dummy, 0);
    if (ret == ACLBLAS_STATUS_INVALID_VALUE) { LOG_PRINT("[SYMV-L2-03] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-03] FAILED\n"); return 1;
}

static uint32_t L204(aclblasHandle h, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, 4, &alpha, &dummy, 3, &dummy, 1, &alpha, &dummy, 1);
    if (ret == ACLBLAS_STATUS_INVALID_VALUE) { LOG_PRINT("[SYMV-L2-04] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-04] FAILED\n"); return 1;
}

static uint32_t L205(aclblasHandle, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 4, &dummy, 1, &alpha, &dummy, 1);
    if (ret == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) { LOG_PRINT("[SYMV-L2-05] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-05] FAILED\n"); return 1;
}

static uint32_t L206(aclblasHandle h, aclrtStream) {
    float alpha = 1.0f, dummy = 0.0f;
    auto ret = aclblasSsymv(h, ACLBLAS_UPPER, 4, nullptr, &dummy, 4, &dummy, 1, &alpha, &dummy, 1);
    if (ret == ACLBLAS_STATUS_INVALID_VALUE) { LOG_PRINT("[SYMV-L2-06] PASSED\n"); return 0; }
    LOG_PRINT("[SYMV-L2-06] FAILED\n"); return 1;
}



#define RUN(fn) do{uint32_t e = fn(handle, stream); tc++; if(e>0) te++;} while(0)

int main()
{
    aclInit(nullptr); aclrtSetDevice(0);
    aclblasHandle handle = nullptr; aclblasCreate(&handle);
    aclrtStream stream = nullptr; aclrtCreateStream(&stream);
    uint32_t tc = 0, te = 0;
    LOG_PRINT("=== SYMV Test Suite (float32) ===\n\n");
    LOG_PRINT("--- L0 ---\n\n");
    RUN(L001); RUN(L002); RUN(L003); RUN(L004); RUN(L005); RUN(L006);
    LOG_PRINT("\n--- L1 ---\n\n");
    RUN(L101); RUN(L102); RUN(L103); RUN(L104); RUN(L105); RUN(L106); RUN(L107); RUN(L108);
    LOG_PRINT("\n--- L2 ---\n\n");
    RUN(L201); RUN(L202); RUN(L203); RUN(L204); RUN(L205); RUN(L206);
    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  Total: %u  Passed: %u  Failed: %u\n", tc, tc - te, te);
    LOG_PRINT("========================================\n");
    if (te == 0) LOG_PRINT("  RESULT: ALL TESTS PASSED\n");
    else LOG_PRINT("  RESULT: %u FAILED\n", te);
    aclrtDestroyStream(stream); aclblasDestroy(handle);
    aclrtResetDevice(0); aclFinalize();
    return te > 0 ? 1 : 0;
}
