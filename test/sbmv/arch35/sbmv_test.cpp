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
 * \file sbmv_test.cpp
 * \brief
 */

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ---- golden / verify / data helpers ----

static std::vector<float> BuildGolden(
    const std::vector<float>& a, const std::vector<float>& x, const std::vector<float>& y, int n, int k, int lda,
    aclblasFillMode uplo, int incx, int incy, float alpha, float beta)
{
    std::vector<float> golden(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        int jStart = (i >= k) ? (i - k) : 0;
        int jEnd = std::min(n, i + k + 1);
        for (int j = jStart; j < jEnd; ++j) {
            int gmIdx;
            if (uplo == ACLBLAS_UPPER)
                gmIdx = (i <= j) ? (k + i - j) + lda * j : (k + j - i) + lda * i;
            else
                gmIdx = (i >= j) ? (i - j) + lda * j : (j - i) + lda * i;
            float xVal = (incx >= 0) ? x[j * incx] : x[(n - 1 - j) * (-incx)];
            acc += a[gmIdx] * xVal;
        }
        float yVal = (incy >= 0) ? y[i * incy] : y[(n - 1 - i) * (-incy)];
        golden[i] = alpha * acc + beta * yVal;
    }
    return golden;
}

static uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden)
{
    std::cout << std::fixed << std::setprecision(6);
    auto printTensor = [](std::vector<float>& t, const char* name) {
        constexpr size_t maxN = 20;
        size_t n = std::min(t.size(), maxN);
        std::cout << name << ": ";
        std::copy(t.begin(), t.begin() + n, std::ostream_iterator<float>(std::cout, " "));
        if (t.size() > maxN)
            std::cout << "...";
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    constexpr float absTol = 1e-3f, relTol = 1e-4f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = std::abs(output[i] - golden[i]);
        float scale = std::max(std::abs(output[i]), std::abs(golden[i]));
        if (diff > absTol && diff > relTol * scale) {
            std::cout << "[Failed] index " << i << " (" << output[i] << " vs " << golden[i] << ")" << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static void FillTestData(
    std::vector<float>& a, std::vector<float>& x, std::vector<float>& y, std::vector<float>& yCopy, int n, int k,
    int lda, aclblasFillMode uplo, int incx, int incy)
{
    size_t xSize = (n > 0) ? static_cast<size_t>(std::abs(incx) * (n - 1) + 1) : 0;
    size_t ySize = (n > 0) ? static_cast<size_t>(std::abs(incy) * (n - 1) + 1) : 0;
    a.resize(static_cast<size_t>(lda) * n, 0.0f);
    x.resize(xSize, 0.0f);
    y.resize(ySize, 0.0f);
    yCopy.resize(ySize, 0.0f);

    std::mt19937 rng(20260515U + static_cast<uint32_t>(n) + static_cast<uint32_t>(k) + static_cast<uint32_t>(uplo));
    std::uniform_real_distribution<float> dist(0.0f, 0.5f);

    for (int j = 0; j < n; ++j)
        for (int row = 0; row < lda; ++row)
            a[row + lda * j] = dist(rng);
    for (size_t i = 0; i < xSize; ++i)
        x[i] = dist(rng);
    for (size_t i = 0; i < ySize; ++i) {
        y[i] = dist(rng);
        yCopy[i] = y[i];
    }
}

static std::vector<float> ExtractYFlat(const std::vector<float>& y, int n, int incy)
{
    std::vector<float> yFlat(n, 0.0f);
    for (int i = 0; i < n; ++i)
        yFlat[i] = (incy >= 0) ? y[i * incy] : y[(n - 1 - i) * (-incy)];
    return yFlat;
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

static int RunCase(
    const char* caseName, int n, int k, aclblasFillMode uplo, float alpha, float beta, int incx, int incy, int lda)
{
    std::cout << "\n[" << caseName << "] n=" << n << " k=" << k
              << " uplo=" << (uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") << " lda=" << lda << " incx=" << incx
              << " incy=" << incy << " alpha=" << alpha << " beta=" << beta << std::endl;

    std::vector<float> a, x, y, yCopy;
    FillTestData(a, x, y, yCopy, n, k, lda, uplo, incx, incy);

    TestContext ctx;
    if (!ctx.Init())
        return -1;

    size_t aSz = static_cast<size_t>(lda) * n * sizeof(float);
    size_t xSz = (n > 0) ? static_cast<size_t>(std::abs(incx) * (n - 1) + 1) * sizeof(float) : 0;
    size_t ySz = (n > 0) ? static_cast<size_t>(std::abs(incy) * (n - 1) + 1) * sizeof(float) : 0;
    if (n > 0 && !ctx.AllocBuffers(a.data(), aSz, x.data(), xSz, y.data(), ySz))
        return -1;

    int ret = aclblasSsbmv(
        ctx.handle, uplo, n, k, &alpha, (const float*)ctx.aDevice, lda, (const float*)ctx.xDevice, incx, &beta,
        (float*)ctx.yDevice, incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSsbmv failed: %d\n", ret);
        return ret;
    }

    if (n > 0) {
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(y.data(), ySz, ctx.yDevice, ySz, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    std::vector<float> yFlat = ExtractYFlat(y, n, incy);
    std::vector<float> golden = BuildGolden(a, x, yCopy, n, k, lda, uplo, incx, incy, alpha, beta);
    int status = VerifyResult(yFlat, golden);
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
    const int n = 10, k = 2, lda = k + 1, incx = 1, incy = 1;
    const float alpha = 1.0f, beta = 0.0f;
    aclblasFillMode uplo = ACLBLAS_LOWER;

    std::vector<float> a(lda * n, 0.0f), x(n, 0.0f), y(n, 0.0f);

    TestContext ctx;
    if (!ctx.Init())
        return -1;

    int failed = 0;
#define CHECK_PARAM(name, expr) failed |= CheckInvalid(name, (expr))

    CHECK_PARAM(
        "invalid uplo", aclblasSsbmv(
                            ctx.handle, static_cast<aclblasFillMode>(100), n, k, &alpha, a.data(), lda, x.data(), incx,
                            &beta, y.data(), incy));
    CHECK_PARAM(
        "k < 0", aclblasSsbmv(ctx.handle, uplo, n, -1, &alpha, a.data(), lda, x.data(), incx, &beta, y.data(), incy));
    CHECK_PARAM(
        "lda < k+1", aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), k, x.data(), incx, &beta, y.data(), incy));
    CHECK_PARAM(
        "incx == 0", aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), lda, x.data(), 0, &beta, y.data(), incy));
    CHECK_PARAM(
        "incy == 0", aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), lda, x.data(), incx, &beta, y.data(), 0));
    CHECK_PARAM(
        "alpha == nullptr",
        aclblasSsbmv(ctx.handle, uplo, n, k, nullptr, a.data(), lda, x.data(), incx, &beta, y.data(), incy));
    CHECK_PARAM(
        "beta == nullptr",
        aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), lda, x.data(), incx, nullptr, y.data(), incy));
    CHECK_PARAM(
        "A == nullptr",
        aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, nullptr, lda, x.data(), incx, &beta, y.data(), incy));
    CHECK_PARAM(
        "x == nullptr",
        aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), lda, nullptr, incx, &beta, y.data(), incy));
    CHECK_PARAM(
        "y == nullptr",
        aclblasSsbmv(ctx.handle, uplo, n, k, &alpha, a.data(), lda, x.data(), incx, &beta, nullptr, incy));
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
    RUN_TC("TC-L0-01", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, 1, 1, 3);
    RUN_TC("TC-L0-02", 32, 3, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1, 4);
    RUN_TC("TC-L0-03", 0, 0, ACLBLAS_LOWER, 1.0f, 0.0f, 1, 1, 1);
    RUN_TC("TC-L0-04", 32, 0, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1, 1);
    RUN_TC("TC-L0-05", 1, 0, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1, 1);
    return fc;
}

static int RunL1Tests()
{
    int fc = 0;
    RUN_TC("TC-L1-01-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, -1, 1, 3);
    RUN_TC("TC-L1-01-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, -1, 1, 3);
    RUN_TC("TC-L1-02-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, 1, -1, 3);
    RUN_TC("TC-L1-02-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, 1, -1, 3);
    RUN_TC("TC-L1-03-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, -1, -1, 3);
    RUN_TC("TC-L1-03-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, -1, -1, 3);
    RUN_TC("TC-L1-04-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, 2, 2, 3);
    RUN_TC("TC-L1-04-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, 2, 2, 3);
    RUN_TC("TC-L1-05-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, -2, -2, 3);
    RUN_TC("TC-L1-05-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, -2, -2, 3);
    RUN_TC("TC-L1-06-U", 32, 2, ACLBLAS_UPPER, 0.8f, 1.2f, 3, 2, 3);
    RUN_TC("TC-L1-06-L", 32, 2, ACLBLAS_LOWER, 0.8f, 1.2f, 3, 2, 3);
    return fc;
}

static int RunGenTests()
{
    int fc = 0;
    struct GenCase {
        int n, k, lda;
        aclblasFillMode uplo;
        float alpha, beta;
        int incx, incy;
    };
    const GenCase cases[] = {
        {0, 0, 1, ACLBLAS_LOWER, 1.0f, 0.0f, 1, 1},
        {1, 0, 1, ACLBLAS_LOWER, 0.5f, 0.5f, -1, -1},
        {13, 0, 1, ACLBLAS_LOWER, 0.5f, 0.5f, 2, 5},
        {100, 20, 25, ACLBLAS_UPPER, 0.0f, 1.5f, 11, 1},
        {1023, 128, 132, ACLBLAS_LOWER, 1.8f, 0.2f, 7, 11},
        {4096, 1, 3, ACLBLAS_UPPER, 0.9f, 0.1f, 5, -3},
        // Large incx=1,incy=1 cases for UB path evaluation
        {1024, 256, 260, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1},
        {4096, 64, 66, ACLBLAS_UPPER, 0.8f, 1.2f, 1, 1},
        {4096, 512, 520, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1},
        // Large incx=1 cases for UB stress testing
        {8192, 1, 3, ACLBLAS_UPPER, 0.9f, 0.1f, 1, 1},
        {8192, 128, 132, ACLBLAS_LOWER, 1.5f, 0.3f, 1, 1},
        {2048, 128, 132, ACLBLAS_UPPER, 1.5f, 0.3f, 1, 1},
    };
    constexpr int kGenCaseCount = sizeof(cases) / sizeof(cases[0]);
    for (int i = 0; i < kGenCaseCount; i++) {
        const GenCase& c = cases[i];
        std::ostringstream nm;
        nm << "TC-GEN-" << std::setw(2) << std::setfill('0') << (i + 1) << "-" << (c.uplo == ACLBLAS_UPPER ? "U" : "L");
        RUN_TC(nm.str().c_str(), c.n, c.k, c.uplo, c.alpha, c.beta, c.incx, c.incy, c.lda);
    }
    return fc;
}

#undef RUN_TC

int32_t main(int32_t argc, char* argv[])
{
    std::cout << "========== SBMV Test ==========" << std::endl;

    std::cout << "\n--- Stage 1: Invalid Parameter Tests ---" << std::endl;
    if (TestInvalidParameters())
        return 1;

    std::cout << "\n--- Stage 2: L0 Functional Tests ---" << std::endl;
    int l0Failed = RunL0Tests();
    std::cout << "\n--- Stage 3: L1 Stride Tests ---" << std::endl;
    int l1Failed = RunL1Tests();
    std::cout << "\n--- Stage 4: Generalization Tests ---" << std::endl;
    int genFailed = RunGenTests();

    int totalL0 = 5, totalL1 = 12, totalGen = 12, totalFailed = l0Failed + l1Failed + genFailed;
    std::cout << "\n========== SBMV Test Complete ==========" << std::endl;
    std::cout << "L0 Results:     " << (totalL0 - l0Failed) << "/" << totalL0 << " passed" << std::endl;
    std::cout << "L1 Results:     " << (totalL1 - l1Failed) << "/" << totalL1 << " passed" << std::endl;
    std::cout << "GEN Results:    " << (totalGen - genFailed) << "/" << totalGen << " passed" << std::endl;
    std::cout << "Total:          " << (totalL0 + totalL1 + totalGen - totalFailed) << "/"
              << (totalL0 + totalL1 + totalGen) << " passed" << std::endl;
    return (totalFailed > 0) ? 1 : 0;
}
