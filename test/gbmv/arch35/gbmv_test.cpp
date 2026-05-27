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
 * \file gbmv_test.cpp
 * \brief ST test for GBMV (General Banded Matrix-Vector multiplication) - FP32.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
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

// ---------------------------------------------------------------------------
// Data helpers (inline, same style as sbmv test)
// ---------------------------------------------------------------------------

static void FillBandedMatrix(float* A, int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t lda, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::fill(A, A + static_cast<size_t>(lda) * static_cast<size_t>(n), 0.0f);
    for (int64_t j = 0; j < n; ++j) {
        int64_t iStart = (j > ku) ? (j - ku) : 0;
        int64_t iEnd = (j + kl < m - 1) ? (j + kl) : (m - 1);
        for (int64_t i = iStart; i <= iEnd; ++i) {
            int64_t bandedRow = ku + i - j;
            A[bandedRow + j * lda] = dist(rng);
        }
    }
}

static float* FillStridedVector(float* v, int64_t len, int64_t inc, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    int64_t absInc = (inc < 0) ? -inc : inc;
    std::fill(v, v + static_cast<size_t>((len - 1) * absInc + 1), 0.0f);
    for (int64_t i = 0; i < len; ++i) {
        v[i * absInc] = dist(rng);
    }
    return (inc < 0) ? (v + (len - 1) * absInc) : v;
}

// ---------------------------------------------------------------------------
// Test case descriptor
// ---------------------------------------------------------------------------
struct GbmvTestCase {
    const char* caseId;
    const char* description;
    aclblasOperation_t trans;
    int64_t m;
    int64_t n;
    int64_t kl;
    int64_t ku;
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
    int64_t lda;
    uint32_t seed;
    bool expectSuccess;
};

// ---------------------------------------------------------------------------
// L0 test case table
// ---------------------------------------------------------------------------
static const GbmvTestCase kTestCasesL0[] = {
    {"TC-L0-001", "Small square banded matrix (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260516,
     true},
    {"TC-L0-002", "alpha=0, beta=1.0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 0.0f, 1.0f, 1, 1, 5, 20260517, true},
    {"TC-L0-003", "beta=0, alpha=1.0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.0f, 1, 1, 5, 20260518, true},
    {"TC-L0-004", "alpha=0, beta=0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 0.0f, 0.0f, 1, 1, 5, 20260519, true},
    {"TC-L0-005", "non-unit incx=2 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.3f, 0.5f, 2, 1, 5, 20260520, true},
    {"TC-L0-006", "non-unit incy=2 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.3f, 0.5f, 1, 2, 5, 20260521, true},
    {"TC-L0-007", "rectangular matrix m!=n (trans=N)", ACLBLAS_OP_N, 10, 6, 2, 3, 1.2f, 0.6f, 1, 1, 6, 20260522, true},
    {"TC-L0-008", "Basic trans=T square matrix", ACLBLAS_OP_T, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260542, true},
    {"TC-L0-009", "trans=T beta=0", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.0f, 1, 1, 5, 20260543, true},
    {"TC-L0-010", "trans=T alpha=0 beta=1 (y unchanged)", ACLBLAS_OP_T, 8, 8, 2, 2, 0.0f, 1.0f, 1, 1, 5, 20260544,
     true},
    {"TC-L0-011", "trans=T alpha=0 beta=0 (zero output)", ACLBLAS_OP_T, 8, 8, 2, 2, 0.0f, 0.0f, 1, 1, 5, 20260545,
     true},
    {"TC-L0-012", "trans=T rectangular m<n", ACLBLAS_OP_T, 6, 10, 2, 3, 1.2f, 0.6f, 1, 1, 6, 20260546, true},
    {"TC-L0-013", "trans=T rectangular m>n", ACLBLAS_OP_T, 10, 6, 3, 2, 1.2f, 0.6f, 1, 1, 6, 20260547, true},
    {"TC-L0-014", "trans=C (same as T for real)", ACLBLAS_OP_C, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260548, true},
};

static constexpr int kNumL0Cases = sizeof(kTestCasesL0) / sizeof(kTestCasesL0[0]);

// ---------------------------------------------------------------------------
// L1 functional test case table
// ---------------------------------------------------------------------------
static const GbmvTestCase kTestCasesL1[] = {
    {"TC-L1-001", "Narrow bandwidth tri-diagonal (kl=ku=1)", ACLBLAS_OP_N, 8, 8, 1, 1, 0.9f, 0.3f, 1, 1, 3, 20260523,
     true},
    {"TC-L1-002", "Wide bandwidth (kl=ku=7) nearly full", ACLBLAS_OP_N, 8, 8, 7, 7, 0.8f, 0.4f, 1, 1, 15, 20260524,
     true},
    {"TC-L1-003", "Near-full matrix (kl=m-1, ku=n-1)", ACLBLAS_OP_N, 8, 8, 7, 7, 1.0f, 0.5f, 1, 1, 15, 20260525, true},
    {"TC-L1-004", "Only lower triangular band (ku=0)", ACLBLAS_OP_N, 8, 8, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260526, true},
    {"TC-L1-005", "Only upper triangular band (kl=0)", ACLBLAS_OP_N, 8, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260527, true},
    {"TC-L1-006", "Asymmetric bandwidth (kl=1, ku=4)", ACLBLAS_OP_N, 8, 8, 1, 4, 1.2f, 0.7f, 1, 1, 6, 20260528, true},
    {"TC-L1-007", "Empty rows m=0 (immediate return)", ACLBLAS_OP_N, 0, 8, 0, 2, 1.0f, 1.0f, 1, 1, 3, 20260529, false},
    {"TC-L1-008", "Empty cols n=0 (immediate return)", ACLBLAS_OP_N, 8, 0, 2, 0, 1.0f, 1.0f, 1, 1, 3, 20260530, false},
    {"TC-L1-009", "kl=0 degenerate (only main+upper diags)", ACLBLAS_OP_N, 8, 8, 0, 3, 1.0f, 0.5f, 1, 1, 4, 20260531,
     true},
    {"TC-L1-010", "ku=0 degenerate (only main+lower diags)", ACLBLAS_OP_N, 8, 8, 3, 0, 1.0f, 0.5f, 1, 1, 4, 20260532,
     true},
    {"TC-L1-011", "kl=ku=0 (main diagonal only)", ACLBLAS_OP_N, 8, 8, 0, 0, 1.0f, 0.5f, 1, 1, 1, 20260533, true},
    {"TC-L1-012", "Negative x stride incx=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, -1, 1, 5, 20260534, true},
    {"TC-L1-013", "Negative y stride incy=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, 1, -1, 5, 20260535, true},
    {"TC-L1-014", "Dual negative strides incx=-1 incy=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, -1, -1, 5, 20260536,
     true},
    {"TC-L1-015", "Non-compact storage lda > kl+ku+1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, 1, 1, 8, 20260537, true},
    {"TC-L1-016", "Large scale (m=n=512, kl=ku=8)", ACLBLAS_OP_N, 512, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17, 20260538, true},
    {"TC-L1-017", "Single row matrix (m=1)", ACLBLAS_OP_N, 1, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260539, true},
    {"TC-L1-018", "Single column matrix (n=1)", ACLBLAS_OP_N, 8, 1, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260540, true},
    {"TC-L1-019", "Zero output large scale (alpha=beta=0)", ACLBLAS_OP_N, 128, 128, 4, 4, 0.0f, 0.0f, 1, 1, 9, 20260541,
     true},

    // --- trans=T / trans=C cases ---
    {"TC-L1-020", "trans=T tri-diagonal (kl=ku=1)", ACLBLAS_OP_T, 8, 8, 1, 1, 0.9f, 0.3f, 1, 1, 3, 20260549, true},
    {"TC-L1-021", "trans=T wide bandwidth (kl=ku=7)", ACLBLAS_OP_T, 8, 8, 7, 7, 0.8f, 0.4f, 1, 1, 15, 20260550, true},
    {"TC-L1-022", "trans=T only lower band (ku=0)", ACLBLAS_OP_T, 8, 8, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260551, true},
    {"TC-L1-023", "trans=T only upper band (kl=0)", ACLBLAS_OP_T, 8, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260552, true},
    {"TC-L1-024", "trans=T kl=ku=0 (main diagonal only)", ACLBLAS_OP_T, 8, 8, 0, 0, 1.0f, 0.5f, 1, 1, 1, 20260553,
     true},
    {"TC-L1-025", "trans=T asymmetric bandwidth (kl=1, ku=4)", ACLBLAS_OP_T, 8, 8, 1, 4, 1.2f, 0.7f, 1, 1, 6, 20260554,
     true},
    {"TC-L1-026", "trans=T non-unit incx=2", ACLBLAS_OP_T, 8, 8, 2, 2, 1.3f, 0.5f, 2, 1, 5, 20260555, true},
    {"TC-L1-027", "trans=T non-unit incy=2", ACLBLAS_OP_T, 8, 8, 2, 2, 1.3f, 0.5f, 1, 2, 5, 20260556, true},
    {"TC-L1-028", "trans=T negative incx=-1", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, -1, 1, 5, 20260557, true},
    {"TC-L1-029", "trans=T negative incy=-1", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, 1, -1, 5, 20260558, true},
    {"TC-L1-030", "trans=T non-compact lda", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, 1, 1, 8, 20260559, true},
    {"TC-L1-031", "trans=T large scale (m=n=512, kl=ku=8)", ACLBLAS_OP_T, 512, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17,
     20260560, true},
    {"TC-L1-032", "trans=T rectangular m<n large scale", ACLBLAS_OP_T, 256, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17, 20260561,
     true},
    {"TC-L1-033", "trans=T single row matrix (m=1)", ACLBLAS_OP_T, 1, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260562, true},
    {"TC-L1-034", "trans=T single column matrix (n=1)", ACLBLAS_OP_T, 8, 1, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260563, true},
    {"TC-L1-035", "trans=T zero output large scale (alpha=beta=0)", ACLBLAS_OP_T, 128, 128, 4, 4, 0.0f, 0.0f, 1, 1, 9,
     20260564, true},
    {"TC-L1-036", "trans=C basic (same as T for real)", ACLBLAS_OP_C, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260565, true},
    {"TC-L1-037", "trans=C large scale", ACLBLAS_OP_C, 128, 128, 4, 4, 1.0f, 0.5f, 1, 1, 9, 20260566, true},
};

static constexpr int kNumL1Cases = sizeof(kTestCasesL1) / sizeof(kTestCasesL1[0]);

// ---------------------------------------------------------------------------
// Precision thresholds
// ---------------------------------------------------------------------------
static constexpr double kFp32Threshold = 1.0 / 8192.0; // 2^(-13)

// ===========================================================================
// Golden reference
// ===========================================================================

static void GbmvGolden(
    aclblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, float alpha, const float* A, int64_t lda,
    const float* x, int64_t incx, float beta, const float* y, int64_t incy, float* yGolden)
{
    int64_t absIncy = (incy < 0) ? -incy : incy;

    if (trans == ACLBLAS_OP_N) {
        for (int64_t i = 0; i < m; ++i) {
            double sum = 0.0;
            int64_t jStart = (i > static_cast<int64_t>(kl)) ? (i - kl) : 0;
            int64_t jEnd = (i + ku < n - 1) ? (i + ku) : (n - 1);
            for (int64_t j = jStart; j <= jEnd; ++j) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[j * incx]);
            }
            yGolden[i * absIncy] = static_cast<float>(
                static_cast<double>(alpha) * sum + static_cast<double>(beta) * static_cast<double>(y[i * incy]));
        }
    } else {
        for (int64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            int64_t firstRow = (j > ku) ? (j - ku) : 0;
            int64_t lastRow = (j + kl < m - 1) ? (j + kl) : (m - 1);
            for (int64_t i = firstRow; i <= lastRow; ++i) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[i * incx]);
            }
            yGolden[j * absIncy] = static_cast<float>(
                static_cast<double>(alpha) * sum + static_cast<double>(beta) * static_cast<double>(y[j * incy]));
        }
    }
}

// ===========================================================================
// Result verification
// ===========================================================================

static void PrintTensors(const float* output, const float* golden, int64_t count, int64_t inc, int64_t absInc,
    const char* caseId)
{
    constexpr size_t kMaxPrintElems = 20;
    std::cout << "[" << caseId << "] Output: ";
    for (size_t k = 0; k < static_cast<size_t>(count) && k < kMaxPrintElems; ++k) {
        std::cout << output[k * inc] << " ";
    }
    if (static_cast<size_t>(count) > kMaxPrintElems) {
        std::cout << "...";
    }
    std::cout << std::endl;

    std::cout << "[" << caseId << "] Golden: ";
    for (size_t k = 0; k < static_cast<size_t>(count) && k < kMaxPrintElems; ++k) {
        std::cout << golden[k * static_cast<size_t>(absInc)] << " ";
    }
    if (static_cast<size_t>(count) > kMaxPrintElems) {
        std::cout << "...";
    }
    std::cout << std::endl;
}

static void ComputeErrorMetrics(const float* output, const float* golden, int64_t count, int64_t inc, int64_t absInc,
    double& sumRelErr, double& maxRelErr, int64_t& outlierCount, double outlierLimit, const char* caseId)
{
    static constexpr double kEpsilon = 1e-7;
    sumRelErr = 0.0;
    maxRelErr = 0.0;
    outlierCount = 0;

    for (int64_t i = 0; i < count; ++i) {
        float outVal = output[i * inc];
        float goldVal = golden[i * absInc];
        double relErr =
            static_cast<double>(std::abs(outVal - goldVal)) / (static_cast<double>(std::abs(goldVal)) + kEpsilon);
        sumRelErr += relErr;
        if (relErr > maxRelErr) {
            maxRelErr = relErr;
        }
        if (relErr > outlierLimit) {
            if (outlierCount < 5) {
                std::cout << "[" << caseId << "] outlier at index " << i << ": kernel=" << outVal
                          << " golden=" << goldVal << " relErr=" << relErr << std::endl;
            }
            ++outlierCount;
        }
    }
}

static uint32_t VerifyResult(const float* output, const float* golden, int64_t count, int64_t inc, const char* caseId)
{
    int64_t absInc = (inc < 0) ? -inc : inc;
    std::cout << std::fixed << std::setprecision(6);
    PrintTensors(output, golden, count, inc, absInc, caseId);

    double sumRelErr, maxRelErr;
    int64_t outlierCount;
    double outlierLimit = 10.0 * kFp32Threshold;
    ComputeErrorMetrics(output, golden, count, inc, absInc, sumRelErr, maxRelErr, outlierCount, outlierLimit, caseId);

    if (outlierCount > 5) {
        std::cout << "[" << caseId << "] ... and " << (outlierCount - 5) << " more outliers" << std::endl;
    }
    double mere = (count > 0) ? sumRelErr / static_cast<double>(count) : 0.0;
    std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr << " (threshold=" << kFp32Threshold
              << ", outlier_limit=" << outlierLimit << ")" << std::endl;

    bool pass = (mere < kFp32Threshold) && (maxRelErr < outlierLimit);
    if (pass) {
        std::cout << "[" << caseId << "] PASSED (MERE < threshold && MARE < 10*threshold, " << outlierCount
                  << " outliers out of " << count << " elements)" << std::endl;
        return 0;
    }
    std::cout << "[" << caseId << "] FAILED (MERE=" << mere << " vs threshold=" << kFp32Threshold
              << ", MARE=" << maxRelErr << " vs limit=" << outlierLimit << ", " << outlierCount << " outliers out of "
              << count << " elements)" << std::endl;
    return 1;
}

// ===========================================================================
// Device lifecycle RAII (same style as sbmv test)
// ===========================================================================

struct TestContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;

    bool Init()
    {
        if (aclInit(nullptr) != ACL_SUCCESS) {
            return false;
        }
        return aclrtSetDevice(deviceId) == ACL_SUCCESS && aclrtCreateStream(&stream) == ACL_SUCCESS &&
               aclblasCreate(&handle) == ACL_SUCCESS && aclblasSetStream(handle, stream) == ACL_SUCCESS;
    }

    bool AllocBuffers(const void* aSrc, size_t aSz, const void* xSrc, size_t xSz, const void* ySrc, size_t ySz)
    {
        aclError r;
        r = aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSz, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, LOG_PRINT("malloc aDevice failed: %d\n", r); return false);
        r = aclrtMalloc(reinterpret_cast<void**>(&xDevice), xSz, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, LOG_PRINT("malloc xDevice failed: %d\n", r);
                  aclrtFree(aDevice); aDevice = nullptr; return false);
        r = aclrtMalloc(reinterpret_cast<void**>(&yDevice), ySz, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(r == ACL_SUCCESS, LOG_PRINT("malloc yDevice failed: %d\n", r);
                  aclrtFree(aDevice); aclrtFree(xDevice); aDevice = xDevice = nullptr; return false);
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
        if (stream) {
            aclrtDestroyStream(stream);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    TestContext() = default;
    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

// ===========================================================================
// Test case runner
// ===========================================================================

static int RunCaseMain(
    const GbmvTestCase& tc, TestContext& ctx, const std::vector<float>& aHost, const std::vector<float>& xHost,
    std::vector<float>& yHost, float* xBlasPtr, float* yBlasPtr, const float* yInitBlasPtr, int64_t xCount,
    int64_t yCount, size_t aBytes, size_t xBytes, size_t yBytes, size_t ySize)
{
    if (!ctx.AllocBuffers(aHost.data(), aBytes, xHost.data(), xBytes, yHost.data(), yBytes)) {
        return -1;
    }
    aclblasStatus_t blasRet = aclblasSgbmv(
        ctx.handle, tc.trans, tc.m, tc.n, tc.kl, tc.ku, &tc.alpha, (const float*)ctx.aDevice, tc.lda,
        (const float*)ctx.xDevice, tc.incx, &tc.beta, (float*)ctx.yDevice, tc.incy);
    CHECK_RET(
        blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("[%s] aclblasSgbmv failed. ERROR: %d\n", tc.caseId, blasRet);
        return blasRet);

    aclError aclRet = aclrtSynchronizeStream(ctx.stream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("[%s] aclrtSynchronizeStream failed. ERROR: %d\n", tc.caseId, aclRet);
        return aclRet);

    aclrtMemcpy(yHost.data(), yBytes, ctx.yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> yGolden(ySize, 0.0f);
    GbmvGolden(
        tc.trans, tc.m, tc.n, tc.kl, tc.ku, tc.alpha, aHost.data(), tc.lda, xBlasPtr, tc.incx, tc.beta, yInitBlasPtr,
        tc.incy, yGolden.data());

    return static_cast<int>(VerifyResult(yBlasPtr, yGolden.data(), yCount, tc.incy, tc.caseId));
}

static int RunCase(const GbmvTestCase& tc)
{
    bool isTransN = (tc.trans == ACLBLAS_OP_N);
    int64_t absIncx = (tc.incx < 0) ? -tc.incx : tc.incx;
    int64_t absIncy = (tc.incy < 0) ? -tc.incy : tc.incy;
    int64_t xCount = isTransN ? tc.n : tc.m;
    int64_t yCount = isTransN ? tc.m : tc.n;

    TestContext ctx;
    if (!ctx.Init()) {
        return -1;
    }

    if (!tc.expectSuccess) {
        size_t dummySz = sizeof(float);
        float dummyVal = 0.0f;
        if (!ctx.AllocBuffers(&dummyVal, dummySz, &dummyVal, dummySz, &dummyVal, dummySz)) {
            return -1;
        }
        aclblasStatus_t blasRet = aclblasSgbmv(
            ctx.handle, tc.trans, tc.m, tc.n, tc.kl, tc.ku, &tc.alpha, (const float*)ctx.aDevice, tc.lda,
            (const float*)ctx.xDevice, tc.incx, &tc.beta, (float*)ctx.yDevice, tc.incy);
        bool earlyReturnOk = (blasRet == ACLBLAS_STATUS_SUCCESS);
        std::cout << "[" << tc.caseId << "] early-return "
                  << (earlyReturnOk ? "OK" : "FAILED (unexpected error)") << " (ret=" << blasRet << ")" << std::endl;
        return earlyReturnOk ? 0 : 1;
    }

    size_t aSize = (tc.n > 0) ? static_cast<size_t>(tc.lda) * static_cast<size_t>(tc.n) : 1;
    size_t xSize = (xCount > 0) ? static_cast<size_t>((xCount - 1) * absIncx + 1) : 1;
    size_t ySize = (yCount > 0) ? static_cast<size_t>((yCount - 1) * absIncy + 1) : 1;
    std::vector<float> aHost(aSize, 0.0f);
    std::vector<float> xHost(xSize, 0.0f);
    std::vector<float> yHost(ySize, 0.0f);

    std::mt19937 rng(tc.seed);
    FillBandedMatrix(aHost.data(), tc.m, tc.n, tc.kl, tc.ku, tc.lda, rng);
    float* xBlasPtr = FillStridedVector(xHost.data(), xCount, tc.incx, rng);
    float* yBlasPtr = FillStridedVector(yHost.data(), yCount, tc.incy, rng);
    std::vector<float> yInitial = yHost;
    float* yInitBlasPtr = (tc.incy < 0) ? yInitial.data() + (yCount - 1) * absIncy : yInitial.data();

    const char* transStr = isTransN ? "N" : (tc.trans == ACLBLAS_OP_T ? "T" : "C");
    std::cout << "\n[" << tc.caseId << "] " << tc.description << " trans=" << transStr << " m=" << tc.m
              << " n=" << tc.n << " kl=" << tc.kl << " ku=" << tc.ku << " lda=" << tc.lda << " incx=" << tc.incx
              << " incy=" << tc.incy << " alpha=" << tc.alpha << " beta=" << tc.beta << std::endl;

    size_t aBytes = aSize * sizeof(float);
    size_t xBytes = xSize * sizeof(float);
    size_t yBytes = ySize * sizeof(float);
    return RunCaseMain(
        tc, ctx, aHost, xHost, yHost, xBlasPtr, yBlasPtr, yInitBlasPtr, xCount, yCount, aBytes, xBytes, yBytes, ySize);
}

// ===========================================================================
// Test suite runner
// ===========================================================================

static void RunTestSuite(
    const GbmvTestCase* cases, int numCases, bool stopOnFailure, int& passed, int& failed, bool& stopped)
{
    for (int i = 0; i < numCases; ++i) {
        int ret = RunCase(cases[i]);
        if (ret == 0) {
            ++passed;
        } else {
            ++failed;
            if (stopOnFailure) {
                std::cout << "\n[FATAL] " << cases[i].caseId << " failed. Stopping." << std::endl;
                stopped = true;
                break;
            }
        }
    }
}

// ===========================================================================
// Main
// ===========================================================================

int32_t main(int32_t argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::cout << "============================================" << std::endl;
    std::cout << "  GBMV ST Test (FP32, experimental_standard)" << std::endl;
    std::cout << "  Precision: FP32 threshold=" << kFp32Threshold << " (2^-13), MERE<threshold && MARE<10*threshold"
              << std::endl;
    std::cout << "  L0 cases: " << kNumL0Cases << ", L1 cases: " << kNumL1Cases << std::endl;
    std::cout << "============================================" << std::endl;

    int passed = 0, failed = 0;
    bool l1Skipped = false;

    std::cout << "\n========== L0 Gate ==========" << std::endl;
    RunTestSuite(kTestCasesL0, kNumL0Cases, true, passed, failed, l1Skipped);

    if (!l1Skipped) {
        std::cout << "\n========== L1 Cases ==========" << std::endl;
        bool dummyStopped = false;
        RunTestSuite(kTestCasesL1, kNumL1Cases, false, passed, failed, dummyStopped);
    } else {
        std::cout << "\n[Skipped] L1 cases not executed due to L0 failure." << std::endl;
    }

    int total = passed + failed;
    std::cout << "\n========== Test Summary ==========" << std::endl;
    std::cout << "Total:  " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    if (total > 0) {
        std::cout << "Pass rate: " << std::fixed << std::setprecision(2) << (100.0 * passed / total) << "%"
                  << std::endl;
    }
    if (l1Skipped) {
        std::cout << "L1 cases: SKIPPED" << std::endl;
    }
    std::cout << "===================================" << std::endl;

    return (failed > 0) ? -1 : 0;
}
