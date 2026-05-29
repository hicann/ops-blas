/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file syr2_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ---- golden / verify / data helpers ----

static inline void GetTriRegion(int row, int n, aclblasFillMode uplo, int& colStart, int& colEnd)
{
    if (uplo == ACLBLAS_UPPER) {
        colStart = row;
        colEnd = n;
    } else {
        colStart = 0;
        colEnd = row + 1;
    }
}

static std::vector<float> BuildGolden(
    const std::vector<float>& a, const std::vector<float>& x, const std::vector<float>& y, int n, int lda,
    aclblasFillMode uplo, int incx, int incy, float alpha)
{
    std::vector<float> golden = a;
    for (int row = 0; row < n; ++row) {
        float xRow = (incx >= 0) ? x[row * incx] : x[(n - 1 - row) * (-incx)];
        float yRow = (incy >= 0) ? y[row * incy] : y[(n - 1 - row) * (-incy)];
        float axRow = alpha * xRow;
        float ayRow = alpha * yRow;

        int colStart, colEnd;
        GetTriRegion(row, n, uplo, colStart, colEnd);
        for (int col = colStart; col < colEnd; ++col) {
            float xCol = (incx >= 0) ? x[col * incx] : x[(n - 1 - col) * (-incx)];
            float yCol = (incy >= 0) ? y[col * incy] : y[(n - 1 - col) * (-incy)];
            golden[row * lda + col] += axRow * yCol + ayRow * xCol;
        }
    }
    return golden;
}

static uint32_t VerifyResult(
    const std::vector<float>& output, const std::vector<float>& golden, int n, int lda, aclblasFillMode uplo)
{
    std::cout << std::fixed << std::setprecision(6);

    constexpr float absTol = 1e-3f;
    constexpr float relTol = 1e-4f;
    uint32_t errors = 0;

    for (int row = 0; row < n; ++row) {
        int colStart, colEnd;
        GetTriRegion(row, n, uplo, colStart, colEnd);
        for (int col = colStart; col < colEnd; ++col) {
            int idx = row * lda + col;
            float diff = std::abs(output[idx] - golden[idx]);
            float scale = std::max(std::abs(output[idx]), std::abs(golden[idx]));
            if (diff > absTol && diff > relTol * scale) {
                if (errors < 10) {
                    std::cout << "[Failed] index (" << row << "," << col << ") out=" << output[idx]
                              << " gold=" << golden[idx] << std::endl;
                }
                ++errors;
            }
        }
    }
    if (errors == 0) {
        std::cout << "[Success] Triangular region verification passed." << std::endl;
    } else {
        std::cout << "[Failed] " << errors << " errors in triangular region." << std::endl;
    }
    return errors;
}

static uint32_t VerifyUnchanged(
    const std::vector<float>& output, const std::vector<float>& original, int n, int lda, aclblasFillMode uplo)
{
    uint32_t errors = 0;
    for (int row = 0; row < n; ++row) {
        int colStart = (uplo == ACLBLAS_UPPER) ? 0 : row + 1;
        int colEnd = (uplo == ACLBLAS_UPPER) ? row : n;
        for (int col = colStart; col < colEnd; ++col) {
            if (output[row * lda + col] != original[row * lda + col]) {
                if (errors < 10) {
                    std::cout << "[Failed] Unchanged region modified at (" << row << "," << col << ")" << std::endl;
                }
                ++errors;
            }
        }
    }
    if (errors == 0) {
        std::cout << "[Success] Unchanged region verification passed." << std::endl;
    } else {
        std::cout << "[Failed] " << errors << " errors in unchanged region." << std::endl;
    }
    return errors;
}

static void FillTestData(
    std::vector<float>& a, std::vector<float>& x, std::vector<float>& y, std::vector<float>& aCopy, int n, int lda,
    aclblasFillMode uplo, int incx, int incy)
{
    size_t xSize = (n > 0) ? static_cast<size_t>(std::abs(incx) * (n - 1) + 1) : 0;
    size_t ySize = (n > 0) ? static_cast<size_t>(std::abs(incy) * (n - 1) + 1) : 0;
    a.resize(static_cast<size_t>(lda) * n, 0.0f);
    x.resize(xSize, 0.0f);
    y.resize(ySize, 0.0f);
    aCopy.resize(static_cast<size_t>(lda) * n, 0.0f);

    std::mt19937 rng(20260515U + static_cast<uint32_t>(n) + static_cast<uint32_t>(lda) + static_cast<uint32_t>(uplo));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
        aCopy[i] = a[i];
    }
    for (size_t i = 0; i < xSize; ++i)
        x[i] = dist(rng);
    for (size_t i = 0; i < ySize; ++i)
        y[i] = dist(rng);
}

// ---- device lifecycle RAII ----

struct TestContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;
    uint8_t *aDevice = nullptr, *xDevice = nullptr, *yDevice = nullptr;

    bool Init()
    {
        if (aclInit(nullptr) != ACL_SUCCESS)
            return false;
        return aclrtSetDevice(deviceId) == ACL_SUCCESS && aclrtCreateStream(&stream) == ACL_SUCCESS &&
               aclblasCreate(&handle) == ACL_SUCCESS && aclblasSetStream(handle, stream) == ACL_SUCCESS;
    }

    bool AllocBuffers(const void* aSrc, size_t aSz, const void* xSrc, size_t xSz, const void* ySrc, size_t ySz)
    {
        aclError r;
        if ((r = aclrtMalloc((void**)&aDevice, aSz, ACL_MEM_MALLOC_HUGE_FIRST)) != ACL_SUCCESS) {
            LOG_PRINT("malloc aDevice failed: %d\n", r);
            return false;
        }
        if ((r = aclrtMalloc((void**)&xDevice, xSz, ACL_MEM_MALLOC_HUGE_FIRST)) != ACL_SUCCESS) {
            LOG_PRINT("malloc xDevice failed: %d\n", r);
            aclrtFree(aDevice);
            aDevice = nullptr;
            return false;
        }
        if ((r = aclrtMalloc((void**)&yDevice, ySz, ACL_MEM_MALLOC_HUGE_FIRST)) != ACL_SUCCESS) {
            LOG_PRINT("malloc yDevice failed: %d\n", r);
            aclrtFree(aDevice);
            aclrtFree(xDevice);
            aDevice = xDevice = nullptr;
            return false;
        }
        aclrtMemcpy(aDevice, aSz, aSrc, aSz, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(xDevice, xSz, xSrc, xSz, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(yDevice, ySz, ySrc, ySz, ACL_MEMCPY_HOST_TO_DEVICE);
        return true;
    }

    ~TestContext()
    {
        aclrtFree(aDevice);
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclblasDestroy(handle);
        if (stream)
            aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    TestContext() = default;
    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

// ---- test runner ----

static int RunCase(const char* caseName, int n, aclblasFillMode uplo, float alpha, int incx, int incy, int lda)
{
    std::cout << "\n[" << caseName << "] n=" << n << " uplo=" << (uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER")
              << " lda=" << lda << " incx=" << incx << " incy=" << incy << " alpha=" << alpha << std::endl;

    std::vector<float> a, x, y, aCopy;
    FillTestData(a, x, y, aCopy, n, lda, uplo, incx, incy);

    TestContext ctx;
    if (!ctx.Init())
        return -1;

    size_t aSz = static_cast<size_t>(lda) * n * sizeof(float);
    size_t xSz = (n > 0) ? static_cast<size_t>(std::abs(incx) * (n - 1) + 1) * sizeof(float) : 0;
    size_t ySz = (n > 0) ? static_cast<size_t>(std::abs(incy) * (n - 1) + 1) * sizeof(float) : 0;
    if (n > 0 && !ctx.AllocBuffers(a.data(), aSz, x.data(), xSz, y.data(), ySz))
        return -1;

    int ret = aclblasSsyr2(
        ctx.handle, uplo, n, &alpha, (const float*)ctx.xDevice, incx, (const float*)ctx.yDevice, incy,
        (float*)ctx.aDevice, lda);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSsyr2 failed: %d\n", ret);
        return ret;
    }

    if (n > 0) {
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(a.data(), aSz, ctx.aDevice, aSz, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    std::vector<float> golden = BuildGolden(aCopy, x, y, n, lda, uplo, incx, incy, alpha);
    uint32_t triErr = VerifyResult(a, golden, n, lda, uplo);
    uint32_t unchErr = VerifyUnchanged(a, aCopy, n, lda, uplo);
    int status = (triErr == 0 && unchErr == 0) ? 0 : 1;
    std::cout << "[" << caseName << "] " << (status == 0 ? "PASSED" : "FAILED") << std::endl;
    return status;
}

// ---- parameter validation ----

static int CheckInvalid(const char* name, int ret)
{
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] " << name << ": expected INVALID_VALUE, got " << ret << std::endl;
        return 1;
    }
    std::cout << "[Success] " << name << " passed." << std::endl;
    return 0;
}

static int TestInvalidParameters()
{
    const int n = 10, lda = 10, incx = 1, incy = 1;
    const float alpha = 1.0f;
    aclblasFillMode uplo = ACLBLAS_LOWER;

    std::vector<float> a(lda * n, 0.0f), x(n, 0.0f), y(n, 0.0f);

    TestContext ctx;
    if (!ctx.Init())
        return -1;

    int failed = 0;
#define CHECK_PARAM(name, expr) failed |= CheckInvalid(name, (expr))

    CHECK_PARAM(
        "invalid uplo",
        aclblasSsyr2(
            ctx.handle, static_cast<aclblasFillMode>(100), n, &alpha, x.data(), incx, y.data(), incy, a.data(), lda));
    CHECK_PARAM("n < 0", aclblasSsyr2(ctx.handle, uplo, -1, &alpha, x.data(), incx, y.data(), incy, a.data(), lda));
    CHECK_PARAM(
        "lda < max(1,n)", aclblasSsyr2(ctx.handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, a.data(), n - 1));
    CHECK_PARAM("incx == 0", aclblasSsyr2(ctx.handle, uplo, n, &alpha, x.data(), 0, y.data(), incy, a.data(), lda));
    CHECK_PARAM("incy == 0", aclblasSsyr2(ctx.handle, uplo, n, &alpha, x.data(), incx, y.data(), 0, a.data(), lda));
    CHECK_PARAM(
        "alpha == nullptr", aclblasSsyr2(ctx.handle, uplo, n, nullptr, x.data(), incx, y.data(), incy, a.data(), lda));
    CHECK_PARAM(
        "x == nullptr", aclblasSsyr2(ctx.handle, uplo, n, &alpha, nullptr, incx, y.data(), incy, a.data(), lda));
    CHECK_PARAM(
        "y == nullptr", aclblasSsyr2(ctx.handle, uplo, n, &alpha, x.data(), incx, nullptr, incy, a.data(), lda));
    CHECK_PARAM(
        "A == nullptr", aclblasSsyr2(ctx.handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, nullptr, lda));
#undef CHECK_PARAM

    if (failed)
        return 1;
    std::cout << "[Success] All parameter validation tests passed." << std::endl;
    return 0;
}

// ---- test suites ----

#define RUN_TC(name, ...)                    \
    do {                                     \
        int _r = RunCase(name, __VA_ARGS__); \
        if (_r != 0)                         \
            fc++;                            \
    } while (0)

static int RunL0Tests()
{
    int fc = 0;
    RUN_TC("TC-L0-01", 32, ACLBLAS_UPPER, 2.0f, 1, 1, 32);
    RUN_TC("TC-L0-02", 32, ACLBLAS_LOWER, 2.0f, 1, 1, 32);
    RUN_TC("TC-L0-03", 64, ACLBLAS_UPPER, 0.0f, 1, 1, 64);
    RUN_TC("TC-L0-04", 64, ACLBLAS_LOWER, 0.0f, 1, 1, 64);
    RUN_TC("TC-L0-05", 1, ACLBLAS_UPPER, 2.0f, 1, 1, 1);
    RUN_TC("TC-L0-06", 1, ACLBLAS_LOWER, 2.0f, 1, 1, 1);
    RUN_TC("TC-L0-07", 64, ACLBLAS_UPPER, -1.5f, 1, 1, 64);
    RUN_TC("TC-L0-08", 64, ACLBLAS_LOWER, -1.5f, 1, 1, 64);
    RUN_TC("TC-L0-09", 32, ACLBLAS_UPPER, 2.0f, 1, 1, 64);
    RUN_TC("TC-L0-10", 32, ACLBLAS_LOWER, 2.0f, 1, 1, 64);
    return fc;
}

static int RunL1Tests()
{
    int fc = 0;
    RUN_TC("TC-L1-01-U", 32, ACLBLAS_UPPER, 2.0f, -1, 1, 32);
    RUN_TC("TC-L1-01-L", 32, ACLBLAS_LOWER, 2.0f, -1, 1, 32);
    RUN_TC("TC-L1-02-U", 32, ACLBLAS_UPPER, 2.0f, 1, -1, 32);
    RUN_TC("TC-L1-02-L", 32, ACLBLAS_LOWER, 2.0f, 1, -1, 32);
    RUN_TC("TC-L1-03-U", 32, ACLBLAS_UPPER, 2.0f, -1, -1, 32);
    RUN_TC("TC-L1-03-L", 32, ACLBLAS_LOWER, 2.0f, -1, -1, 32);
    RUN_TC("TC-L1-04-U", 32, ACLBLAS_UPPER, 2.0f, 2, 3, 32);
    RUN_TC("TC-L1-04-L", 32, ACLBLAS_LOWER, 2.0f, 2, 3, 32);
    RUN_TC("TC-L1-05-U", 32, ACLBLAS_UPPER, 2.0f, -2, -3, 32);
    RUN_TC("TC-L1-05-L", 32, ACLBLAS_LOWER, 2.0f, -2, -3, 32);
    RUN_TC("TC-L1-06-U", 64, ACLBLAS_UPPER, 0.5f, 3, 2, 64);
    RUN_TC("TC-L1-06-L", 64, ACLBLAS_LOWER, 0.5f, 3, 2, 64);
    return fc;
}

static int RunGenTests()
{
    int fc = 0;
    struct GenCase {
        int n, lda;
        aclblasFillMode uplo;
        float alpha;
        int incx, incy;
    };
    const GenCase cases[] = {
        {0, 1, ACLBLAS_LOWER, 1.0f, 1, 1},
        {1, 1, ACLBLAS_LOWER, 0.5f, -1, -1},
        {13, 13, ACLBLAS_LOWER, 0.5f, 2, 5},
        {100, 100, ACLBLAS_UPPER, 0.0f, 11, 1},
        {1023, 1024, ACLBLAS_LOWER, 1.8f, 7, 11},
        {4096, 4096, ACLBLAS_UPPER, 0.9f, 5, -3},
        // Large incx=1,incy=1 cases for UB path evaluation
        {1024, 1024, ACLBLAS_LOWER, 0.8f, 1, 1},
        {4096, 4096, ACLBLAS_UPPER, 0.8f, 1, 1},
        {4096, 4100, ACLBLAS_LOWER, 0.8f, 1, 1},
        // Large incx=1 cases for UB stress testing
        {8192, 8192, ACLBLAS_UPPER, 0.9f, 1, 1},
        {8192, 8192, ACLBLAS_LOWER, 1.5f, 1, 1},
        {2048, 2050, ACLBLAS_UPPER, 1.5f, 1, 1},
    };
    constexpr int kGenCaseCount = sizeof(cases) / sizeof(cases[0]);
    for (int i = 0; i < kGenCaseCount; i++) {
        const GenCase& c = cases[i];
        std::ostringstream nm;
        nm << "TC-GEN-" << std::setw(2) << std::setfill('0') << (i + 1) << "-" << (c.uplo == ACLBLAS_UPPER ? "U" : "L");
        RUN_TC(nm.str().c_str(), c.n, c.uplo, c.alpha, c.incx, c.incy, c.lda);
    }
    return fc;
}

#undef RUN_TC

int32_t main(int32_t argc, char* argv[])
{
    std::cout << "========== SYR2 Test ==========" << std::endl;

    std::cout << "\n--- Stage 1: Invalid Parameter Tests ---" << std::endl;
    if (TestInvalidParameters())
        return 1;

    std::cout << "\n--- Stage 2: L0 Functional Tests ---" << std::endl;
    int l0Failed = RunL0Tests();
    std::cout << "\n--- Stage 3: L1 Stride Tests ---" << std::endl;
    int l1Failed = RunL1Tests();
    std::cout << "\n--- Stage 4: Generalization Tests ---" << std::endl;
    int genFailed = RunGenTests();

    int totalL0 = 10, totalL1 = 12, totalGen = 12, totalFailed = l0Failed + l1Failed + genFailed;
    std::cout << "\n========== SYR2 Test Complete ==========" << std::endl;
    std::cout << "L0 Results:     " << (totalL0 - l0Failed) << "/" << totalL0 << " passed" << std::endl;
    std::cout << "L1 Results:     " << (totalL1 - l1Failed) << "/" << totalL1 << " passed" << std::endl;
    std::cout << "GEN Results:    " << (totalGen - genFailed) << "/" << totalGen << " passed" << std::endl;
    std::cout << "Total:          " << (totalL0 + totalL1 + totalGen - totalFailed) << "/"
              << (totalL0 + totalL1 + totalGen) << " passed" << std::endl;
    return (totalFailed > 0) ? 1 : 0;
}
