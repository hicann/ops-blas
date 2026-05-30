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
 * \file sspmv_test.cpp
 * \brief ST test for aclblasSspmv (ascend950 SIMT VF)
 */

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

using namespace std;

constexpr float ATOL = 1e-3f;
constexpr float RTOL = 1e-4f;

// ====================================================================
// BuildGolden: CPU reference result for y = alpha * A * x + beta * y
// ====================================================================
static vector<float> BuildGolden(
    const vector<float>& aPacked, const vector<float>& x, const vector<float>& yOrig, int n, aclblasFillMode uplo,
    int incx, int incy, float alpha, float beta)
{
    vector<float> golden(static_cast<size_t>(n), 0.0f);

    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;

        for (int j = 0; j < n; ++j) {
            // Resolve A(i,j) from packed storage via symmetry
            size_t pIdx;
            if (uplo == ACLBLAS_UPPER) {
                if (i <= j) {
                    pIdx = static_cast<size_t>(j) * (j + 1) / 2 + static_cast<size_t>(i);
                } else {
                    pIdx = static_cast<size_t>(i) * (i + 1) / 2 + static_cast<size_t>(j);
                }
            } else { // ACLBLAS_LOWER
                if (i >= j) {
                    pIdx =
                        static_cast<size_t>(j) * n - static_cast<size_t>(j) * (j - 1) / 2 + static_cast<size_t>(i - j);
                } else {
                    pIdx =
                        static_cast<size_t>(i) * n - static_cast<size_t>(i) * (i - 1) / 2 + static_cast<size_t>(j - i);
                }
            }

            // x with stride
            float xVal;
            if (incx >= 0) {
                xVal = x[j * incx];
            } else {
                xVal = x[(n - 1 - j) * (-incx)];
            }
            acc += aPacked[pIdx] * xVal;
        }

        // y with stride (from original y, before kernel modification)
        float yVal;
        if (incy >= 0) {
            yVal = yOrig[i * incy];
        } else {
            yVal = yOrig[(n - 1 - i) * (-incy)];
        }
        golden[static_cast<size_t>(i)] = alpha * acc + beta * yVal;
    }

    return golden;
}

// ====================================================================
// VerifyResult: element-wise comparison with atol/rtol
// ====================================================================
static int VerifyResult(vector<float>& output, vector<float>& golden, const char* caseName)
{
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = fabs(output[i] - golden[i]);
        float scale = fmax(fabs(output[i]), fabs(golden[i]));
        if (diff > ATOL && diff > RTOL * scale) {
            cout << "[Failed] " << caseName << ": accuracy failed at index " << i << " (output=" << output[i]
                 << " golden=" << golden[i] << " diff=" << diff << ")" << endl;
            return 1;
        }
    }
    cout << "[Success] " << caseName << ": accuracy passed." << endl;
    return 0;
}

// ====================================================================
// FillTestData: random data with deterministic seed (no ACL dependency)
// ====================================================================
static void FillTestData(
    vector<float>& aPacked, vector<float>& x, vector<float>& y, vector<float>& yCopy, int n, aclblasFillMode uplo,
    int incx, int incy)
{
    size_t apSize = static_cast<size_t>(n) * (n + 1) / 2;
    size_t xSize = (n > 0) ? static_cast<size_t>(abs(incx) * (n - 1) + 1) : 0;
    size_t ySize = (n > 0) ? static_cast<size_t>(abs(incy) * (n - 1) + 1) : 0;

    aPacked.assign(apSize, 0.0f);
    x.assign(xSize, 0.0f);
    y.assign(ySize, 0.0f);
    yCopy.assign(ySize, 0.0f);

    mt19937 rng(20260521U + static_cast<uint32_t>(n) + static_cast<uint32_t>(uplo));
    uniform_real_distribution<float> dist(0.0f, 0.5f);

    // Fill only the stored triangle of the packed symmetric matrix
    for (int j = 0; j < n; ++j) {
        if (uplo == ACLBLAS_UPPER) {
            for (int i = 0; i <= j; ++i) {
                size_t idx = static_cast<size_t>(j) * (j + 1) / 2 + static_cast<size_t>(i);
                aPacked[idx] = dist(rng);
            }
        } else {
            for (int i = j; i < n; ++i) {
                size_t idx =
                    static_cast<size_t>(j) * n - static_cast<size_t>(j) * (j - 1) / 2 + static_cast<size_t>(i - j);
                aPacked[idx] = dist(rng);
            }
        }
    }

    for (size_t i = 0; i < xSize; ++i)
        x[i] = dist(rng);
    for (size_t i = 0; i < ySize; ++i) {
        y[i] = dist(rng);
        yCopy[i] = y[i]; // save for golden beta*y computation
    }
}

// ====================================================================
// ExtractYFlat: de-stride kernel output into contiguous vector
// ====================================================================
static vector<float> ExtractYFlat(const vector<float>& y, int n, int incy)
{
    vector<float> yFlat(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        if (incy >= 0) {
            yFlat[static_cast<size_t>(i)] = y[i * incy];
        } else {
            yFlat[static_cast<size_t>(i)] = y[(n - 1 - i) * (-incy)];
        }
    }
    return yFlat;
}

// ====================================================================
// TestContext: RAII device lifecycle management
// ====================================================================
struct TestContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;
    uint8_t* apDevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;

    bool Init()
    {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            cerr << "[Failed] aclInit: " << ret << endl;
            return false;
        }
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            cerr << "[Failed] aclrtSetDevice: " << ret << endl;
            return false;
        }
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            cerr << "[Failed] aclrtCreateStream: " << ret << endl;
            return false;
        }
        aclblasStatus_t st = aclblasCreate(reinterpret_cast<aclblasHandle_t*>(&handle));
        if (st != ACLBLAS_STATUS_SUCCESS) {
            cerr << "[Failed] aclblasCreate: " << st << endl;
            return false;
        }
        st = aclblasSetStream(handle, stream);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            cerr << "[Failed] aclblasSetStream: " << st << endl;
            return false;
        }
        return true;
    }

    bool AllocBuffers(const vector<float>& aPacked, const vector<float>& xVec, const vector<float>& yVec)
    {
        auto doMalloc = [](uint8_t*& dev, size_t bytes, const void* host) -> bool {
            if (bytes == 0)
                return true;
            aclError ret = aclrtMalloc(reinterpret_cast<void**>(&dev), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                cerr << "[Failed] aclrtMalloc: " << ret << endl;
                return false;
            }
            ret = aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                cerr << "[Failed] aclrtMemcpy H2D: " << ret << endl;
                return false;
            }
            return true;
        };

        size_t apBytes = aPacked.size() * sizeof(float);
        size_t xBytes = xVec.size() * sizeof(float);
        size_t yBytes = yVec.size() * sizeof(float);

        return doMalloc(apDevice, apBytes, aPacked.data()) && doMalloc(xDevice, xBytes, xVec.data()) &&
               doMalloc(yDevice, yBytes, yVec.data());
    }

    // Convenience accessors for passing to aclblasSspmv
    const float* apFloat() const { return reinterpret_cast<const float*>(apDevice); }
    const float* xFloat() const { return reinterpret_cast<const float*>(xDevice); }
    float* yFloat() { return reinterpret_cast<float*>(yDevice); }

    ~TestContext()
    {
        if (apDevice)
            aclrtFree(apDevice);
        if (xDevice)
            aclrtFree(xDevice);
        if (yDevice)
            aclrtFree(yDevice);
        if (handle)
            aclblasDestroy(handle);
        if (stream)
            aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

// ====================================================================
// RunCase: single test case (data prep -> API call -> verify)
// ====================================================================
static int RunCase(const char* caseName, int n, aclblasFillMode uplo, float alpha, float beta, int incx, int incy)
{
    vector<float> aPacked, x, y, yCopy;
    FillTestData(aPacked, x, y, yCopy, n, uplo, incx, incy);

    TestContext ctx;
    if (!ctx.Init()) {
        cerr << "[Failed] " << caseName << ": context init failed." << endl;
        return 1;
    }

    // n == 0: Host returns SUCCESS without launching kernel
    if (n == 0) {
        int ret = aclblasSspmv(
            ctx.handle, uplo, 0, &alpha, reinterpret_cast<const float*>(ctx.apDevice),
            reinterpret_cast<const float*>(ctx.xDevice), incx, &beta, reinterpret_cast<float*>(ctx.yDevice), incy);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            cout << "[Failed] " << caseName << ": n=0 expected SUCCESS, got " << ret << endl;
            return 1;
        }
        cout << "[Success] " << caseName << ": n=0 returned SUCCESS." << endl;
        return 0;
    }

    if (!ctx.AllocBuffers(aPacked, x, y)) {
        cerr << "[Failed] " << caseName << ": buffer alloc failed." << endl;
        return 1;
    }

    int ret = aclblasSspmv(ctx.handle, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        cout << "[Failed] " << caseName << ": aclblasSspmv returned " << ret << endl;
        return 1;
    }

    aclrtSynchronizeStream(ctx.stream);
    size_t yBytes = y.size() * sizeof(float);
    aclrtMemcpy(y.data(), yBytes, ctx.yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    vector<float> yFlat = ExtractYFlat(y, n, incy);
    vector<float> golden = BuildGolden(aPacked, x, yCopy, n, uplo, incx, incy, alpha, beta);
    return VerifyResult(yFlat, golden, caseName);
}

// ====================================================================
// Check helpers for parameter validation
// ====================================================================
static int CheckInvalid(const char* name, aclblasStatus_t ret)
{
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        cout << "[Failed] " << name << ": expected INVALID_VALUE(3), got " << ret << endl;
        return 1;
    }
    cout << "[Success] " << name << endl;
    return 0;
}

static int CheckHandleInvalid(const char* name, aclblasStatus_t ret)
{
    if (ret != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        cout << "[Failed] " << name << ": expected HANDLE_IS_NULLPTR(9), got " << ret << endl;
        return 1;
    }
    cout << "[Success] " << name << endl;
    return 0;
}

// ====================================================================
// Stage 1: Parameter Validation (10 cases)
// ====================================================================
static int TestInvalidParameters()
{
    cout << "\n=== Stage 1: Parameter Validation ===" << endl;
    int failed = 0;
    const int n = 10, incx = 1, incy = 1;
    const float alpha = 1.0f, beta = 0.0f;
    aclblasFillMode uplo = ACLBLAS_LOWER;
    vector<float> aPacked(n * (n + 1) / 2, 0.0f);
    vector<float> xV(static_cast<size_t>(n), 0.0f);
    vector<float> yV(static_cast<size_t>(n), 0.0f);

    TestContext ctx;
    if (!ctx.Init() || !ctx.AllocBuffers(aPacked, xV, yV)) {
        cerr << "[Failed] TestInvalidParameters: setup failed." << endl;
        return 1;
    }
    auto api = [&](aclblasHandle h, aclblasFillMode u, int nn, const float* a, const float* ap, const float* xp, int ix,
                   const float* b, float* yp, int iy) { return aclblasSspmv(h, u, nn, a, ap, xp, ix, b, yp, iy); };
    failed += CheckInvalid(
        "TC-IV-01 invalid uplo",
        api(ctx.handle, (aclblasFillMode)100, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-02 n=-1",
        api(ctx.handle, uplo, -1, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-03 incx=0", api(ctx.handle, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), 0, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-04 incy=0", api(ctx.handle, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), 0));
    failed += CheckInvalid(
        "TC-IV-05 alpha=nullptr",
        api(ctx.handle, uplo, n, nullptr, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-06 beta=nullptr",
        api(ctx.handle, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, nullptr, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-07 AP=nullptr",
        api(ctx.handle, uplo, n, &alpha, nullptr, ctx.xFloat(), incx, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-08 x=nullptr",
        api(ctx.handle, uplo, n, &alpha, ctx.apFloat(), nullptr, incx, &beta, ctx.yFloat(), incy));
    failed += CheckInvalid(
        "TC-IV-09 y=nullptr",
        api(ctx.handle, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, nullptr, incy));
    failed += CheckHandleInvalid(
        "TC-IV-10 handle=nullptr",
        api(nullptr, uplo, n, &alpha, ctx.apFloat(), ctx.xFloat(), incx, &beta, ctx.yFloat(), incy));

    return failed;
}

// ====================================================================
// Stage 2: L0 Basic Functionality (5 cases)
// ====================================================================
static int RunL0Tests()
{
    cout << "\n=== Stage 2: L0 Basic Functionality ===" << endl;
    int failed = 0;

    failed += RunCase("TC-L0-01 UPPER n=4", 4, ACLBLAS_UPPER, 0.8f, 1.2f, 1, 1);
    failed += RunCase("TC-L0-02 LOWER n=4", 4, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1);
    failed += RunCase("TC-L0-03 n=0", 0, ACLBLAS_LOWER, 1.0f, 0.0f, 1, 1);
    failed += RunCase("TC-L0-04 n=1", 1, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1);
    failed += RunCase("TC-L0-05 n=128", 128, ACLBLAS_LOWER, 0.8f, 1.2f, 1, 1);

    return failed;
}

// ====================================================================
// Stage 3: GEN Randomized Cases (5 permanent cases)
// Values hand-picked to cover various n sizes up to 4096,
// mixed uplo, varied strides (positive/negative/large), and edge scalars.
// ====================================================================
static int RunGenTests()
{
    cout << "\n=== Stage 3: GEN Randomized Cases ===" << endl;
    struct GenCase {
        int n;
        aclblasFillMode uplo;
        float alpha, beta;
        int incx, incy;
    };
    const GenCase cases[] = {
        {1024, ACLBLAS_UPPER, 1.7f, 0.3f, 3, 5},  {2048, ACLBLAS_LOWER, 0.0f, 2.0f, -2, 1},
        {4096, ACLBLAS_UPPER, 0.5f, 0.5f, 1, -3}, {512, ACLBLAS_LOWER, 2.3f, 1.1f, -5, -2},
        {3072, ACLBLAS_UPPER, 0.1f, 0.9f, 7, 4},
    };
    int failed = 0;
    for (const auto& c : cases) {
        const char* tag = (c.uplo == ACLBLAS_UPPER) ? "U" : "L";
        ostringstream oss;
        oss << "TC-GEN n=" << c.n << " " << tag << " incx=" << c.incx << " incy=" << c.incy;
        failed += RunCase(oss.str().c_str(), c.n, c.uplo, c.alpha, c.beta, c.incx, c.incy);
    }
    return failed;
}

// ====================================================================
// main
// ====================================================================
int main(int argc, char* argv[])
{
    cout << "=== SPMV ST Test (ascend950) ===" << endl;

    int failed = 0;
    failed += TestInvalidParameters();
    failed += RunL0Tests();
    failed += RunGenTests();

    cout << "\n=== Summary ===" << endl;
    cout << "Stage 1 (Parameter Validation) : 10 cases" << endl;
    cout << "Stage 2 (L0 Basic Functionality):  5 cases" << endl;
    cout << "Stage 3 (GEN Randomized)       :  5 cases" << endl;
    if (failed == 0) {
        cout << "RESULT: ALL PASSED" << endl;
        return 0;
    }
    cout << "RESULT: " << failed << " FAILURE(S)" << endl;
    return 1;
}
