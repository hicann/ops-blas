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
*
* Tests the aclblasSgbmv host API against a CPU double-precision golden reference.
* Precision verification follows experimental_standard.md:
*   MERE = avg(|actual - golden| / (|golden| + 1e-7))
*   MARE = max(|actual - golden| / (|golden| + 1e-7))
*   Pass: MERE < 2^-13 && MARE < 10 * 2^-13
* L0 test cases (14): trans=N (7) + trans=T/C (7), covering basic functionality,
*                     alpha/beta edge cases, non-unit strides, rectangular matrices.
* L1 test cases (37): trans=N (19) + trans=T/C (18), covering bandwidth variants,
*                     edge shapes, negative strides, non-compact lda, large scale,
*                     zero-output verification.
*/

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "gbmv_test_utils.h"

// ---------------------------------------------------------------------------
// Test case descriptor
// ---------------------------------------------------------------------------
struct GbmvTestCase {
    const char *caseId;
    const char *description;
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
    {"TC-L0-001", "Small square banded matrix (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260516, true},
    {"TC-L0-002", "alpha=0, beta=1.0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 0.0f, 1.0f, 1, 1, 5, 20260517, true},
    {"TC-L0-003", "beta=0, alpha=1.0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.0f, 1, 1, 5, 20260518, true},
    {"TC-L0-004", "alpha=0, beta=0 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 0.0f, 0.0f, 1, 1, 5, 20260519, true},
    {"TC-L0-005", "non-unit incx=2 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.3f, 0.5f, 2, 1, 5, 20260520, true},
    {"TC-L0-006", "non-unit incy=2 (trans=N)", ACLBLAS_OP_N, 8, 8, 2, 2, 1.3f, 0.5f, 1, 2, 5, 20260521, true},
    {"TC-L0-007", "rectangular matrix m!=n (trans=N)", ACLBLAS_OP_N, 10, 6, 2, 3, 1.2f, 0.6f, 1, 1, 6, 20260522, true},
    {"TC-L0-008", "Basic trans=T square matrix", ACLBLAS_OP_T, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260542, true},
    {"TC-L0-009", "trans=T beta=0", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.0f, 1, 1, 5, 20260543, true},
    {"TC-L0-010", "trans=T alpha=0 beta=1 (y unchanged)", ACLBLAS_OP_T, 8, 8, 2, 2, 0.0f, 1.0f, 1, 1, 5, 20260544, true},
    {"TC-L0-011", "trans=T alpha=0 beta=0 (zero output)", ACLBLAS_OP_T, 8, 8, 2, 2, 0.0f, 0.0f, 1, 1, 5, 20260545, true},
    {"TC-L0-012", "trans=T rectangular m<n", ACLBLAS_OP_T, 6, 10, 2, 3, 1.2f, 0.6f, 1, 1, 6, 20260546, true},
    {"TC-L0-013", "trans=T rectangular m>n", ACLBLAS_OP_T, 10, 6, 3, 2, 1.2f, 0.6f, 1, 1, 6, 20260547, true},
    {"TC-L0-014", "trans=C (same as T for real)", ACLBLAS_OP_C, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260548, true},
};

static constexpr int kNumL0Cases = sizeof(kTestCasesL0) / sizeof(kTestCasesL0[0]);

// ---------------------------------------------------------------------------
// L1 functional test case table
// ---------------------------------------------------------------------------
static const GbmvTestCase kTestCasesL1[] = {
    {"TC-L1-001", "Narrow bandwidth tri-diagonal (kl=ku=1)", ACLBLAS_OP_N, 8, 8, 1, 1, 0.9f, 0.3f, 1, 1, 3, 20260523, true},
    {"TC-L1-002", "Wide bandwidth (kl=ku=7) nearly full", ACLBLAS_OP_N, 8, 8, 7, 7, 0.8f, 0.4f, 1, 1, 15, 20260524, true},
    {"TC-L1-003", "Near-full matrix (kl=m-1, ku=n-1)", ACLBLAS_OP_N, 8, 8, 7, 7, 1.0f, 0.5f, 1, 1, 15, 20260525, true},
    {"TC-L1-004", "Only lower triangular band (ku=0)", ACLBLAS_OP_N, 8, 8, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260526, true},
    {"TC-L1-005", "Only upper triangular band (kl=0)", ACLBLAS_OP_N, 8, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260527, true},
    {"TC-L1-006", "Asymmetric bandwidth (kl=1, ku=4)", ACLBLAS_OP_N, 8, 8, 1, 4, 1.2f, 0.7f, 1, 1, 6, 20260528, true},
    {"TC-L1-007", "Empty rows m=0 (immediate return)", ACLBLAS_OP_N, 0, 8, 0, 2, 1.0f, 1.0f, 1, 1, 3, 20260529, false},
    {"TC-L1-008", "Empty cols n=0 (immediate return)", ACLBLAS_OP_N, 8, 0, 2, 0, 1.0f, 1.0f, 1, 1, 3, 20260530, false},
    {"TC-L1-009", "kl=0 degenerate (only main+upper diags)", ACLBLAS_OP_N, 8, 8, 0, 3, 1.0f, 0.5f, 1, 1, 4, 20260531, true},
    {"TC-L1-010", "ku=0 degenerate (only main+lower diags)", ACLBLAS_OP_N, 8, 8, 3, 0, 1.0f, 0.5f, 1, 1, 4, 20260532, true},
    {"TC-L1-011", "kl=ku=0 (main diagonal only)", ACLBLAS_OP_N, 8, 8, 0, 0, 1.0f, 0.5f, 1, 1, 1, 20260533, true},
    {"TC-L1-012", "Negative x stride incx=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, -1, 1, 5, 20260534, true},
    {"TC-L1-013", "Negative y stride incy=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, 1, -1, 5, 20260535, true},
    {"TC-L1-014", "Dual negative strides incx=-1 incy=-1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, -1, -1, 5, 20260536, true},
    {"TC-L1-015", "Non-compact storage lda > kl+ku+1", ACLBLAS_OP_N, 8, 8, 2, 2, 1.0f, 0.5f, 1, 1, 8, 20260537, true},
    {"TC-L1-016", "Large scale (m=n=512, kl=ku=8)", ACLBLAS_OP_N, 512, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17, 20260538, true},
    {"TC-L1-017", "Single row matrix (m=1)", ACLBLAS_OP_N, 1, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260539, true},
    {"TC-L1-018", "Single column matrix (n=1)", ACLBLAS_OP_N, 8, 1, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260540, true},
    {"TC-L1-019", "Zero output large scale (alpha=beta=0)", ACLBLAS_OP_N, 128, 128, 4, 4, 0.0f, 0.0f, 1, 1, 9, 20260541, true},

    // --- trans=T / trans=C cases (exercises ProcessTransT + ReduceSum path) ---
    {"TC-L1-020", "trans=T tri-diagonal (kl=ku=1)", ACLBLAS_OP_T, 8, 8, 1, 1, 0.9f, 0.3f, 1, 1, 3, 20260549, true},
    {"TC-L1-021", "trans=T wide bandwidth (kl=ku=7)", ACLBLAS_OP_T, 8, 8, 7, 7, 0.8f, 0.4f, 1, 1, 15, 20260550, true},
    {"TC-L1-022", "trans=T only lower band (ku=0)", ACLBLAS_OP_T, 8, 8, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260551, true},
    {"TC-L1-023", "trans=T only upper band (kl=0)", ACLBLAS_OP_T, 8, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260552, true},
    {"TC-L1-024", "trans=T kl=ku=0 (main diagonal only)", ACLBLAS_OP_T, 8, 8, 0, 0, 1.0f, 0.5f, 1, 1, 1, 20260553, true},
    {"TC-L1-025", "trans=T asymmetric bandwidth (kl=1, ku=4)", ACLBLAS_OP_T, 8, 8, 1, 4, 1.2f, 0.7f, 1, 1, 6, 20260554, true},
    {"TC-L1-026", "trans=T non-unit incx=2", ACLBLAS_OP_T, 8, 8, 2, 2, 1.3f, 0.5f, 2, 1, 5, 20260555, true},
    {"TC-L1-027", "trans=T non-unit incy=2", ACLBLAS_OP_T, 8, 8, 2, 2, 1.3f, 0.5f, 1, 2, 5, 20260556, true},
    {"TC-L1-028", "trans=T negative incx=-1", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, -1, 1, 5, 20260557, true},
    {"TC-L1-029", "trans=T negative incy=-1", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, 1, -1, 5, 20260558, true},
    {"TC-L1-030", "trans=T non-compact lda", ACLBLAS_OP_T, 8, 8, 2, 2, 1.0f, 0.5f, 1, 1, 8, 20260559, true},
    {"TC-L1-031", "trans=T large scale (m=n=512, kl=ku=8)", ACLBLAS_OP_T, 512, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17, 20260560, true},
    {"TC-L1-032", "trans=T rectangular m<n large scale", ACLBLAS_OP_T, 256, 512, 8, 8, 1.0f, 0.5f, 1, 1, 17, 20260561, true},
    {"TC-L1-033", "trans=T single row matrix (m=1)", ACLBLAS_OP_T, 1, 8, 0, 4, 1.0f, 0.5f, 1, 1, 5, 20260562, true},
    {"TC-L1-034", "trans=T single column matrix (n=1)", ACLBLAS_OP_T, 8, 1, 4, 0, 1.0f, 0.5f, 1, 1, 5, 20260563, true},
    {"TC-L1-035", "trans=T zero output large scale (alpha=beta=0)", ACLBLAS_OP_T, 128, 128, 4, 4, 0.0f, 0.0f, 1, 1, 9, 20260564, true},
    {"TC-L1-036", "trans=C basic (same as T for real)", ACLBLAS_OP_C, 8, 8, 2, 2, 1.5f, 0.8f, 1, 1, 5, 20260565, true},
    {"TC-L1-037", "trans=C large scale", ACLBLAS_OP_C, 128, 128, 4, 4, 1.0f, 0.5f, 1, 1, 9, 20260566, true},
};

static constexpr int kNumL1Cases = sizeof(kTestCasesL1) / sizeof(kTestCasesL1[0]);

// ---------------------------------------------------------------------------
// Precision thresholds (per experimental_standard.md)
// FP32: threshold = 2^(-13), pass condition: MERE < threshold && MARE < 10*threshold
// ---------------------------------------------------------------------------
static constexpr double kFp32Threshold = 1.0 / 8192.0;  // 2^(-13)

// ===========================================================================
// Golden reference
// ===========================================================================

static void gbmv_golden(aclblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
    float alpha, const float *A, int64_t lda,
    const float *x, int64_t incx, float beta, const float *y, int64_t incy,
    float *yGolden)
{
    int64_t absIncy = (incy < 0) ? -incy : incy;

    if (trans == ACLBLAS_OP_N) {
        for (int64_t i = 0; i < m; i++) {
            double sum = 0.0;
            int64_t jStart = (i > static_cast<int64_t>(kl)) ? (i - kl) : 0;
            int64_t jEnd = (i + ku < n - 1) ? (i + ku) : (n - 1);
            for (int64_t j = jStart; j <= jEnd; j++) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[j * incx]);
            }
            yGolden[i * absIncy] = static_cast<float>(static_cast<double>(alpha) * sum +
                static_cast<double>(beta) * static_cast<double>(y[i * incy]));
        }
    } else {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            int64_t firstRow = (j > ku) ? (j - ku) : 0;
            int64_t lastRow = (j + kl < m - 1) ? (j + kl) : (m - 1);
            for (int64_t i = firstRow; i <= lastRow; i++) {
                int64_t bandIdx = (ku + i - j) + j * lda;
                sum += static_cast<double>(A[bandIdx]) * static_cast<double>(x[i * incx]);
            }
            yGolden[j * absIncy] = static_cast<float>(static_cast<double>(alpha) * sum +
                static_cast<double>(beta) * static_cast<double>(y[j * incy]));
        }
    }
}

// ===========================================================================
// Result verification
// ===========================================================================

static uint32_t verify_result(const float *output, const float *golden, int64_t m, int64_t incy,
    const char *caseId)
{
    int64_t absIncy = (incy < 0) ? -incy : incy;

    std::cout << std::fixed << std::setprecision(6);

    constexpr size_t kMaxPrintElems = 20;
    std::cout << "[" << caseId << "] Output: ";
    for (size_t k = 0; k < static_cast<size_t>(m) && k < kMaxPrintElems; k++) {
        std::cout << output[k * incy] << " ";
    }
    if (static_cast<size_t>(m) > kMaxPrintElems) std::cout << "...";
    std::cout << std::endl;

    std::cout << "[" << caseId << "] Golden: ";
    for (size_t k = 0; k < static_cast<size_t>(m) && k < kMaxPrintElems; k++) {
        std::cout << golden[k * static_cast<size_t>(absIncy)] << " ";
    }
    if (static_cast<size_t>(m) > kMaxPrintElems) std::cout << "...";
    std::cout << std::endl;

    // Compute MERE (Mean Relative Error) and MARE (Max Relative Error)
    // per experimental_standard.md:
    //   relErr = |actual - golden| / (|golden| + 1e-7)
    double sumRelErr = 0.0;
    double maxRelErr = 0.0;
    int64_t outlierCount = 0;
    static constexpr double kEpsilon = 1e-7;
    double outlierLimit = 10.0 * kFp32Threshold;

    for (int64_t i = 0; i < m; i++) {
        float outVal = output[i * incy];
        float goldVal = golden[i * absIncy];
        double relErr = static_cast<double>(std::abs(outVal - goldVal)) /
                        (static_cast<double>(std::abs(goldVal)) + kEpsilon);
        sumRelErr += relErr;
        if (relErr > maxRelErr) maxRelErr = relErr;

        if (relErr > outlierLimit) {
            if (outlierCount < 5) {
                std::cout << "[" << caseId << "] outlier at index " << i << ": kernel=" << outVal
                          << " golden=" << goldVal << " relErr=" << relErr << std::endl;
            }
            outlierCount++;
        }
    }
    if (outlierCount > 5) {
        std::cout << "[" << caseId << "] ... and " << (outlierCount - 5) << " more outliers" << std::endl;
    }

    double mere = (m > 0) ? sumRelErr / static_cast<double>(m) : 0.0;

    std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr
              << " (threshold=" << kFp32Threshold << ", outlier_limit=" << outlierLimit << ")"
              << std::endl;

    bool pass = (mere < kFp32Threshold) && (maxRelErr < outlierLimit);

    if (pass) {
        std::cout << "[" << caseId << "] PASSED (MERE < threshold && MARE < 10*threshold, "
                  << outlierCount << " outliers out of " << m << " elements)" << std::endl;
        return 0;
    }
    std::cout << "[" << caseId << "] FAILED (MERE=" << mere << " vs threshold=" << kFp32Threshold
              << ", MARE=" << maxRelErr << " vs limit=" << outlierLimit
              << ", " << outlierCount << " outliers out of " << m << " elements)" << std::endl;
    return 1;
}

// ===========================================================================
// ACL lifecycle helpers
// ===========================================================================

static int SetupAclForTest(int32_t deviceId, aclrtStream &stream, aclblasHandle_t &handle)
{
    aclError aclRet = aclInit(nullptr);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtSetDevice(deviceId);
    CHECK_RET(aclRet == ACL_SUCCESS,
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", aclRet); aclFinalize(); return aclRet);

    aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", aclRet);
        aclrtResetDevice(deviceId); aclFinalize(); return aclRet);

    aclblasStatus_t blasRet = aclblasCreate(&handle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
        LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
        aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return blasRet);

    blasRet = aclblasSetStream(handle, stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
        LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
        aclblasDestroy(handle); aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
        return blasRet);

    return 0;
}

static void CleanupAclForTest(aclblasHandle_t handle, aclrtStream stream, int32_t deviceId)
{
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ===========================================================================
// Test case runner
// ===========================================================================

static int run_test_case(const GbmvTestCase &tc)
{
    bool isTransN = (tc.trans == ACLBLAS_OP_N);
    const char *transStr = isTransN ? "N" : (tc.trans == ACLBLAS_OP_T ? "T" : "C");
    int32_t deviceId = 0;
    int64_t absIncx = (tc.incx < 0) ? -tc.incx : tc.incx;
    int64_t absIncy = (tc.incy < 0) ? -tc.incy : tc.incy;

    int64_t xCount = isTransN ? tc.n : tc.m;
    int64_t yCount = isTransN ? tc.m : tc.n;

    size_t aSize = (tc.n > 0) ? static_cast<size_t>(tc.lda) * static_cast<size_t>(tc.n) : 1;
    std::vector<float> aHost(aSize, 0.0f);
    size_t xSize = (xCount > 0) ? static_cast<size_t>((xCount - 1) * absIncx + 1) : 1;
    std::vector<float> xHost(xSize, 0.0f);
    size_t ySize = (yCount > 0) ? static_cast<size_t>((yCount - 1) * absIncy + 1) : 1;
    std::vector<float> yHost(ySize, 0.0f);

    std::mt19937 rng(tc.seed);
    fill_banded_matrix(aHost.data(), tc.m, tc.n, tc.kl, tc.ku, tc.lda, rng);
    float *xBlasPtr = fill_strided_vector(xHost.data(), xCount, tc.incx, rng);
    float *yBlasPtr = fill_strided_vector(yHost.data(), yCount, tc.incy, rng);
    std::vector<float> yInitial = yHost;
    float *yInitBlasPtr = (tc.incy < 0) ? yInitial.data() + (yCount - 1) * absIncy : yInitial.data();

    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    int setupRet = SetupAclForTest(deviceId, stream, handle);
    if (setupRet != 0) return setupRet;

    aclblasStatus_t blasRet = aclblasSgbmv(handle, tc.trans, tc.m, tc.n, tc.kl, tc.ku, &tc.alpha,
        aHost.data(), tc.lda, xBlasPtr, tc.incx, &tc.beta, yBlasPtr, tc.incy);

    if (!tc.expectSuccess) {
        bool earlyReturnOk = (blasRet == ACLBLAS_STATUS_SUCCESS);
        std::cout << "[" << tc.caseId << "] early-return "
                  << (earlyReturnOk ? "OK" : "FAILED (unexpected error)")
                  << " (ret=" << blasRet << ")" << std::endl;
        CleanupAclForTest(handle, stream, deviceId);
        return earlyReturnOk ? 0 : 1;
    }

    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
        LOG_PRINT("[%s] aclblasSgbmv failed. ERROR: %d\n", tc.caseId, blasRet);
        CleanupAclForTest(handle, stream, deviceId); return blasRet);

    aclError aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
        LOG_PRINT("[%s] aclrtSynchronizeStream failed. ERROR: %d\n", tc.caseId, aclRet);
        CleanupAclForTest(handle, stream, deviceId); return aclRet);

    std::vector<float> yGolden(ySize, 0.0f);
    gbmv_golden(tc.trans, tc.m, tc.n, tc.kl, tc.ku, tc.alpha, aHost.data(), tc.lda,
        xBlasPtr, tc.incx, tc.beta, yInitBlasPtr, tc.incy, yGolden.data());

    int status = static_cast<int>(verify_result(yBlasPtr, yGolden.data(), yCount, tc.incy, tc.caseId));
    CleanupAclForTest(handle, stream, deviceId);
    return status;
}

// ===========================================================================
// Test suite runner
// ===========================================================================

static void RunTestSuite(const GbmvTestCase *cases, int numCases, bool stopOnFailure,
                          int &passed, int &failed, bool &stopped)
{
    for (int i = 0; i < numCases; i++) {
        int ret = run_test_case(cases[i]);
        if (ret == 0) {
            passed++;
        } else {
            failed++;
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

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::cout << "============================================" << std::endl;
    std::cout << "  GBMV ST Test (FP32, experimental_standard)" << std::endl;
    std::cout << "  Precision: FP32 threshold=" << kFp32Threshold
              << " (2^-13), MERE<threshold && MARE<10*threshold" << std::endl;
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
        std::cout << "Pass rate: " << std::fixed << std::setprecision(2)
                  << (100.0 * passed / total) << "%" << std::endl;
    }
    if (l1Skipped) std::cout << "L1 cases: SKIPPED" << std::endl;
    std::cout << "===================================" << std::endl;

    return (failed > 0) ? -1 : 0;
}
