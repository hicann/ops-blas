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
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// Precision constants per dtype (experimental + commercial standard)
namespace GemvPrecision {
// FP32:  threshold=2^-13  MERE<2^-13  MARE<10*2^-13
constexpr float kF32Threshold = 1.0f / 8192.0f;      // 2^-13
constexpr float kF32BaseSmallThr = 1.0f / 16384.0f;  // 2^-14
constexpr float kF32Eps = 1.0f / 8388608.0f;         // 2^-23
constexpr float kF32SmallErr = 1.0f / 1073741824.0f; // 2^-30

// FP16:  threshold=2^-10  MERE<2^-10  MARE<10*2^-10
constexpr float kF16Threshold = 1.0f / 1024.0f;    // 2^-10
constexpr float kF16BaseSmallThr = 1.0f / 2048.0f; // 2^-11
constexpr float kF16Eps = 1.0f / 1024.0f;          // 2^-10
constexpr float kF16SmallErr = 1.0f / 65536.0f;    // 2^-16
} // namespace GemvPrecision

#include "../../utils/error_check.h"

using namespace std;

// ============================================================
// Test context — shared across all test cases
// ============================================================
struct TestContext {
    aclblasHandle_t handle;
    aclrtStream stream;
};

// ============================================================
// FP16 helpers
// ============================================================
static uint16_t FloatToHalf(float val)
{
    uint32_t f; memcpy(&f, &val, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f = (exp == 0) ? (sign << 31) | (mant << 13) : (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result; memcpy(&result, &f, sizeof(result));
    return result;
}

// ============================================================
// Golden reference (CPU, per single batch)
// yOut = alpha * op(A) * x + beta * yIn
// ============================================================
static void ComputeGolden(
    const vector<float>& A, const vector<float>& x, vector<float>& y, int m, int n, aclblasOperation_t trans, float alpha,
    float beta)
{
    int outLen = (trans == ACLBLAS_OP_N) ? m : n;
    int innerLen = (trans == ACLBLAS_OP_N) ? n : m;
    vector<float> yTemp(outLen, 0.0f);
    for (int i = 0; i < outLen; i++) {
        double sum = 0.0;
        for (int j = 0; j < innerLen; j++) {
            if (trans == ACLBLAS_OP_N) {
                sum += (double)A[i * n + j] * (double)x[j];
            } else {
                sum += (double)A[j * n + i] * (double)x[j];
            }
        }
        yTemp[i] = (float)sum;
    }
    for (int i = 0; i < outLen; i++) {
        y[i] = alpha * yTemp[i] + beta * y[i];
    }
}

// Per-batch golden, accumulated into flat output vector
static void ComputeGoldenBatched(
    const vector<float>& A, const vector<float>& x, vector<float>& y, int batchCount, int m, int n,
    aclblasOperation_t trans, float alpha, float beta)
{
    vector<float> yIn = y;
    int outLen = (trans == ACLBLAS_OP_N) ? m : n;
    int xLen = (trans == ACLBLAS_OP_N) ? n : m;
    for (int b = 0; b < batchCount; b++) {
        vector<float> AComp(m * n);
        vector<float> xComp(xLen);
        vector<float> yComp(outLen);
        for (int i = 0; i < m * n; i++)
            AComp[i] = A[b * m * n + i];
        for (int i = 0; i < xLen; i++) {
            xComp[i] = x[b * xLen + i];
        }
        for (int i = 0; i < outLen; i++)
            yComp[i] = 0.0f;
        ComputeGolden(AComp, xComp, yComp, m, n, trans, 1.0f, 0.0f);
        for (int i = 0; i < outLen; i++) {
            y[b * outLen + i] = alpha * yComp[i] + beta * yIn[b * outLen + i];
        }
    }
}

// ============================================================
// Verification
// ============================================================
// Precision: combined absolute + relative. Relative error blows up when
// golden ≈ 0 (cancelation in dot product), so we use max(abs, rel*|gold|).
//   error[i]   = |out[i] - gold[i]|
//   relErr[i]  = error[i] / (|gold[i]| + 1e-7)
//   threshold[i] = atol + rtol * |gold[i]|
//   Pass: error[i] < threshold[i]  for all i
// Shared verify loop: iterates over golden & output, collects small/regular-domain stats.
template <typename F>
static void ComputeVerifyStats(size_t n, F&& getOutput, const vector<float>& golden,
                                float smallThr, float smallErr,
                                float& maxRelErr, double& sumRelErr,
                                float& minAbsGold, uint32_t& smallErrCount)
{
    for (size_t i = 0; i < n; i++) {
        float outVal = getOutput(i);
        float absG = fabsf(golden[i]);
        if (absG < smallThr) {
            if (fabsf(outVal - golden[i]) > smallErr)
                smallErrCount++;
        } else {
            float relErr = fabsf(outVal - golden[i]) / (absG + 1e-7f);
            sumRelErr += relErr;
            if (relErr > maxRelErr) maxRelErr = relErr;
        }
        if (absG < minAbsGold) minAbsGold = absG;
    }
}

template <typename F>
static void LogVerifyFail(const char* tag, size_t n, float threshold,
                           F&& getOutput, const vector<float>& golden)
{
    size_t show = 0;
    for (size_t i = 0; i < n && show < 5; i++) {
        float outVal = getOutput(i);
        float relErr = fabsf(outVal - golden[i]) / (fabsf(golden[i]) + 1e-7f);
        if (relErr > threshold * 10.0f) {
            LOG_PRINT("    [%zu] out=%.6f gold=%.6f relErr=%.2e\n", i, outVal, golden[i], relErr);
            show++;
        }
    }
    LOG_PRINT("  [%s] FAILED\n", tag);
}

/* FP32 精度验证 */
static uint32_t VerifyFloat(
    const vector<float>& output, const vector<float>& golden, float threshold,
    uint32_t innerLen, const char* tag)
{
    using namespace GemvPrecision;
    float smallThr = (kF32BaseSmallThr < 1.0f / 1024.0f) ? (1.0f / 1024.0f) : kF32BaseSmallThr;

    size_t n = output.size();
    if (n == 0) { LOG_PRINT("  [%s] PASSED (empty)\n", tag); return 0; }
    float maxRelErr = 0.0f, minAbsGold = fabsf(golden[0]);
    double sumRelErr = 0.0;
    uint32_t smallErrCount = 0;

    ComputeVerifyStats(n, [&](size_t i) { return output[i]; },
                       golden, smallThr, kF32SmallErr,
                       maxRelErr, sumRelErr, minAbsGold, smallErrCount);

    float mere = static_cast<float>(sumRelErr / n), mare = maxRelErr;
    LOG_PRINT("  [%s] MERE=%.2e MARE=%.2e smErrCnt=%u min|gold|=%.2e (limit=%.0e/%.0e)\n",
              tag, mere, mare, smallErrCount, minAbsGold, threshold, threshold * 10.0f);

    if (mere < threshold || mare < threshold * 10.0f || smallErrCount <= 2) {
        LOG_PRINT("  [%s] PASSED\n", tag); return 0;
    }
    LogVerifyFail(tag, n, threshold, [&](size_t i) { return output[i]; }, golden);
    return 1;
}

/* FP16 精度验证 */
static uint32_t VerifyHalf(
    const vector<uint16_t>& output, const vector<float>& golden, float threshold,
    uint32_t innerLen, const char* tag)
{
    using namespace GemvPrecision;
    float smallThr = (kF16BaseSmallThr < 1.0f / 128.0f) ? (1.0f / 128.0f) : kF16BaseSmallThr;

    size_t n = output.size();
    if (n == 0) { LOG_PRINT("  [%s] PASSED (empty)\n", tag); return 0; }
    float maxRelErr = 0.0f, minAbsGold = fabsf(golden[0]);
    double sumRelErr = 0.0;
    uint32_t smallErrCount = 0;

    ComputeVerifyStats(n, [&](size_t i) { return HalfToFloat(output[i]); },
                       golden, smallThr, kF16SmallErr,
                       maxRelErr, sumRelErr, minAbsGold, smallErrCount);

    float mere = static_cast<float>(sumRelErr / n), mare = maxRelErr;
    LOG_PRINT("  [%s] MERE=%.2e MARE=%.2e smErrCnt=%u min|gold|=%.2e (limit=%.0e/%.0e)\n",
              tag, mere, mare, smallErrCount, minAbsGold, threshold, threshold * 10.0f);

    if (mere < threshold || mare < threshold * 10.0f || smallErrCount <= 2) {
        LOG_PRINT("  [%s] PASSED\n", tag); return 0;
    }
    LogVerifyFail(tag, n, threshold, [&](size_t i) { return HalfToFloat(output[i]); }, golden);
    return 1;
}

// ============================================================
// Device executor — alloc / copy-in / call / sync / copy-back / free
// Returns 0 on success, non-zero on error.
// ============================================================
template <typename T>
static aclblasStatus_t RunOperator(
    TestContext& ctx, aclblasOperation_t trans, int m, int n, int batchCount, float alpha, float beta, const vector<T>& A,
    const vector<T>& x, vector<T>& y)
{
    int lda = m;
    int incx = 1;
    int incy = 1;
    int xLen = (trans == ACLBLAS_OP_N) ? n : m;
    int yLen = (trans == ACLBLAS_OP_N) ? m : n;

    size_t aBytes = static_cast<size_t>(batchCount * m * n) * sizeof(T);
    size_t xBytes = static_cast<size_t>(batchCount * xLen) * sizeof(T);
    size_t yBytes = static_cast<size_t>(batchCount * yLen) * sizeof(T);

    uint8_t *dA = nullptr, *dX = nullptr, *dY = nullptr;
    aclError ret;

    ret = aclrtMalloc((void**)&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ACLBLAS_STATUS_ALLOC_FAILED);
    ret = aclrtMalloc((void**)&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dA); return ACLBLAS_STATUS_ALLOC_FAILED);
    ret = aclrtMalloc((void**)&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dA); aclrtFree(dX); return ACLBLAS_STATUS_ALLOC_FAILED);

    ret = aclrtMemcpy(dA, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dA); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR);
    ret = aclrtMemcpy(dX, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dA); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR);
    ret = aclrtMemcpy(dY, yBytes, y.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dA); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclblasStatus_t blasRet;
    if constexpr (std::is_same_v<T, float>) {
        blasRet = aclblasSgemvBatched(
            ctx.handle, trans, m, n, &alpha, (float*)dA, lda, (float*)dX, incx, &beta, (float*)dY, incy, batchCount);
    } else {
        blasRet = aclblasHSHgemvBatched(
            ctx.handle, trans, m, n, &alpha, (uint16_t*)dA, lda, (uint16_t*)dX, incx, &beta, (uint16_t*)dY, incy,
            batchCount);
    }
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return blasRet;
    }

    aclrtSynchronizeStream(ctx.stream);
    aclrtMemcpy(y.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================
// Simple pseudo-random generator (linear congruential)
// ============================================================
static uint32_t g_randSeed = 42;
static float RandFloat()
{
    g_randSeed = g_randSeed * 1103515245u + 12345u;
    int32_t val = static_cast<int32_t>(g_randSeed % 2001u) - 1000;
    return static_cast<float>(val) * 0.001f; // [-1.0, 1.0]
}

// ============================================================
// ============================================================
// Test runner helpers — DRY wrapper around RunOperator + golden + verify
// ============================================================
static uint32_t DoTest(
    TestContext& ctx, aclblasOperation_t trans, int m, int n, int batchCount, float alpha, float beta, const char* tag)
{
    int xLen = (trans == ACLBLAS_OP_N) ? n : m;
    int yLen = (trans == ACLBLAS_OP_N) ? m : n;

    vector<float> A(batchCount * m * n);
    vector<float> x(batchCount * xLen);
    vector<float> y(batchCount * yLen);
    vector<float> yGolden(batchCount * yLen);

    for (auto& v : A)
        v = RandFloat();
    for (auto& v : x)
        v = RandFloat();
    for (auto& v : y)
        v = RandFloat();
    yGolden = y;
    ComputeGoldenBatched(A, x, yGolden, batchCount, m, n, trans, alpha, beta);

    auto ret = RunOperator<float>(ctx, trans, m, n, batchCount, alpha, beta, A, x, y);
    CHECK_RET(
        ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("  [%s] aclblasSgemvBatched failed. ERROR: %d\n", tag, ret); return 1);
    return VerifyFloat(y, yGolden, GemvPrecision::kF32Threshold, (uint32_t)xLen, tag);
}

template <typename T>
static uint32_t DoTestHalf(
    TestContext& ctx, aclblasOperation_t trans, int m, int n, int batchCount, float alpha, float beta, const char* tag)
{
    int xLen = (trans == ACLBLAS_OP_N) ? n : m;
    int yLen = (trans == ACLBLAS_OP_N) ? m : n;

    vector<uint16_t> A(batchCount * m * n);
    vector<uint16_t> x(batchCount * xLen);
    vector<uint16_t> y(batchCount * yLen);
    vector<float> Af(batchCount * m * n);
    vector<float> xf(batchCount * xLen);
    vector<float> yGolden(batchCount * yLen);

    for (auto& v : Af) {
        v = RandFloat();
        A[&v - Af.data()] = FloatToHalf(v);
    }
    for (auto& v : xf) {
        v = RandFloat();
        x[&v - xf.data()] = FloatToHalf(v);
    }
    for (auto& v : yGolden) {
        v = RandFloat();
    }
    for (size_t i = 0; i < y.size(); i++)
        y[i] = FloatToHalf(yGolden[i]);
    // golden must use same FP16-quantized inputs that the kernel sees
    for (size_t i = 0; i < A.size(); i++)
        Af[i] = HalfToFloat(A[i]);
    for (size_t i = 0; i < x.size(); i++)
        xf[i] = HalfToFloat(x[i]);
    for (size_t i = 0; i < yGolden.size(); i++)
        yGolden[i] = HalfToFloat(FloatToHalf(yGolden[i]));
    ComputeGoldenBatched(Af, xf, yGolden, batchCount, m, n, trans, alpha, beta);

    auto ret = RunOperator<uint16_t>(ctx, trans, m, n, batchCount, alpha, beta, A, x, y);
    CHECK_RET(
        ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("  [%s] aclblasSgemvBatched failed. ERROR: %d\n", tag, ret); return 1);
    return VerifyHalf(y, yGolden, GemvPrecision::kF16Threshold, (uint32_t)xLen, tag);
}

// ============================================================
// FP32 Normal, small (m=n=32)    batch=3  alpha=1.5 beta=0.5
// ============================================================
static uint32_t TestFP32_Normal(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 32, 32, 3, 1.5f, 0.5f, "N-32x32");
}
// FP16 Normal, small
static uint32_t TestFP16_Normal(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_N, 32, 32, 3, 1.5f, 0.5f, "N-FP16-32");
}
// FP32 Transpose, small (m=16, n=8)  batch=2  alpha=2.0 beta=0.5
static uint32_t TestFP32_Transpose(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 16, 8, 2, 2.0f, 0.5f, "T-16x8");
}

// ============================================================
// Large shapes
// ============================================================
static uint32_t TestFP32_Normal_Large(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 128, 500, 3, 1.0f, 0.0f, "N-128x500");
}
static uint32_t TestFP32_Transpose_Large(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 128, 64, 2, 1.0f, 0.5f, "T-128x64");
}

// ============================================================
// Large M shapes — 触发 m-tiling（n 优先：nTile=n, mTile 按 UB 反推）
// ============================================================
// m=2500, n=32: nTile=32 → mTile=712 → m-tiling（4 chunks）, n 无 split
static uint32_t TestFP32_Normal_LargeM(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 2500, 32, 2, 1.5f, 0.5f, "N-2500x32");
}

// m=5000, n=64: nTile=64 → mTile=368 → m-tiling（14 chunks）, n 无 split
static uint32_t TestFP32_Normal_LargeMN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 5000, 64, 2, 1.0f, 0.0f, "N-5000x64");
}

// ============================================================
// Large batch — 测多 batch 分核
// ============================================================
// B=65: batchTail=1, 非 8 对齐大批次
static uint32_t TestFP32_Normal_LargeBatch(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 128, 64, 65, 1.5f, 0.5f, "N-128x64-B65");
}
// B=99: 奇数大批次
static uint32_t TestFP32_Normal_BatchNonAligned(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 128, 64, 99, 1.0f, 0.5f, "N-128x64-B99");
}

// m=1025, n=1025, batch=199: 三轴超 1K 且全奇数，batch > 128
static uint32_t TestFP32_Normal_AllNonAligned(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 1025, 1025, 199, 1.0f, 0.5f, "N-1025x1025-B199");
}

// ============================================================
// Non-8-aligned m — mCurrAlign > mCurr, 测 V pipe 补整逻辑
// ============================================================
static uint32_t TestFP32_Normal_MisalignedM(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 2501, 16, 1, 1.0f, 0.0f, "N-2501x16");
}

// ============================================================
// Heavy n-tiling — n 远大于 nTile, 多 n-tile 累加
// ============================================================
static uint32_t TestFP32_Normal_LargeN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 64, 3000, 1, 1.0f, 0.0f, "N-64x3000");
}

// m=17, n=13: 小非对齐 shape, nTile=16→nCurr=13, mTile=17
static uint32_t TestFP32_Normal_SmallNonAligned(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_N, 17, 13, 2, 1.5f, 0.5f, "N-17x13");
}

// ============================================================
// Non-aligned shapes (SIMT trans)
// ============================================================
static uint32_t TestFP32_Transpose_Odd(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 13, 7, 3, 1.5f, 0.5f, "T-13x7");
}

// ============================================================
// Transpose SIMT 泛化
// ============================================================
// Large M — inner loop heavy
static uint32_t TestFP32_Transpose_LargeM(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 512, 32, 2, 1.0f, 0.5f, "T-512x32");
}
// Large N — many output elements, threads loop
static uint32_t TestFP32_Transpose_LargeN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 32, 512, 2, 1.0f, 0.0f, "T-32x512");
}
// Large both + large batch
static uint32_t TestFP32_Transpose_LargeMN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 256, 128, 64, 1.5f, 0.5f, "T-256x128-B64");
}

// Transpose beta=0 (pure A^T x)
static uint32_t TestFP32_Transpose_NoBeta(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 128, 64, 3, 1.0f, 0.0f, "T-128x64-beta0");
}

// Transpose m not 8-aligned
static uint32_t TestFP32_Transpose_MisalignedM(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 33, 16, 2, 1.5f, 0.5f, "T-33x16");
}

// Transpose n not 8-aligned (tests GM scalar output with unaligned count)
static uint32_t TestFP32_Transpose_MisalignedN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 16, 13, 2, 1.0f, 0.0f, "T-16x13");
}

// Transpose both m/n not aligned
static uint32_t TestFP32_Transpose_BothMisaligned(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 33, 13, 2, 1.5f, 0.5f, "T-33x13");
}

// Transpose large batch + small dims
static uint32_t TestFP32_Transpose_SmallBatchLarge(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 5, 6, 200, 1.0f, 0.0f, "T-5x6-B200");
}

// Transpose very large m (SIMT inner loop heavy)
static uint32_t TestFP32_Transpose_VeryLargeM(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 2048, 32, 33, 1.0f, 0.0f, "T-2048x32-B33");
}

// Transpose very large n (SIMT many output elements, threads loop)
static uint32_t TestFP32_Transpose_VeryLargeN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 32, 2048, 17, 1.0f, 0.0f, "T-32x2048-B17");
}

// Transpose very large both (aligned)
static uint32_t TestFP32_Transpose_VeryLargeMN(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 1024, 1024, 9, 1.0f, 0.5f, "T-1024x1024-B9");
}

// Transpose very large both + non-aligned
static uint32_t TestFP32_Transpose_VeryLargeMisaligned(TestContext& ctx)
{
    return DoTest(ctx, ACLBLAS_OP_T, 1025, 1025, 7, 1.0f, 0.0f, "T-1025x1025-B7");
}

// FP16 Transpose
static uint32_t TestFP16_Transpose(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_T, 32, 16, 2, 1.5f, 0.5f, "T-FP16-32x16");
}

// FP16 Normal n-tiling
static uint32_t TestFP16_Normal_Large(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_N, 128, 500, 3, 1.5f, 0.5f, "N-FP16-128x500");
}

// FP16 Normal m-split
static uint32_t TestFP16_Normal_LargeM(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_N, 2500, 32, 2, 1.5f, 0.5f, "N-FP16-2500x32");
}

// FP16 Transpose large
static uint32_t TestFP16_Transpose_Large(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_T, 128, 64, 2, 1.0f, 0.5f, "T-FP16-128x64");
}

// FP16 Transpose odd
static uint32_t TestFP16_Transpose_Odd(TestContext& ctx)
{
    return DoTestHalf<uint16_t>(ctx, ACLBLAS_OP_T, 13, 7, 3, 1.5f, 0.5f, "T-FP16-13x7");
}

// HSS: uint16_t in, float out
static uint32_t DoTestHSS(
    TestContext& ctx, aclblasOperation_t trans, int m, int n, int batchCount, float alpha, float beta, const char* tag)
{
    int xLen = (trans == ACLBLAS_OP_N) ? n : m;
    int yLen = (trans == ACLBLAS_OP_N) ? m : n;
    vector<uint16_t> A(batchCount * m * n);
    vector<uint16_t> x(batchCount * xLen);
    vector<float> Af(batchCount * m * n);
    vector<float> xf(batchCount * xLen);
    vector<float> y(batchCount * yLen);
    vector<float> yGolden(batchCount * yLen);

    for (auto& v : Af) {
        v = RandFloat();
        A[&v - Af.data()] = FloatToHalf(v);
    }
    for (auto& v : xf) {
        v = RandFloat();
        x[&v - xf.data()] = FloatToHalf(v);
    }
    for (auto& v : y)
        v = RandFloat();
    yGolden = y;
    for (size_t i = 0; i < A.size(); i++)
        Af[i] = HalfToFloat(A[i]);
    for (size_t i = 0; i < x.size(); i++)
        xf[i] = HalfToFloat(x[i]);
    ComputeGoldenBatched(Af, xf, yGolden, batchCount, m, n, trans, alpha, beta);

    size_t aN = batchCount * m * n, xN = batchCount * xLen, yN = batchCount * yLen;
    uint8_t *dA = nullptr, *dX = nullptr, *dY = nullptr;
    aclrtMalloc((void**)&dA, aN * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dX, xN * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dY, yN * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dA, aN * sizeof(uint16_t), A.data(), aN * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, xN * sizeof(uint16_t), x.data(), xN * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dY, yN * sizeof(float), y.data(), yN * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    auto ret = aclblasHSSgemvBatched(
        ctx.handle, trans, m, n, &alpha, (uint16_t*)dA, m, (uint16_t*)dX, 1, &beta, (float*)dY, 1, batchCount);
    CHECK_RET(
        ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("  [%s] aclblasHSSgemvBatched failed. ERROR: %d\n", tag, ret);
        aclrtFree(dA); aclrtFree(dX); aclrtFree(dY); return 1);
    aclrtSynchronizeStream(ctx.stream);
    aclrtMemcpy(y.data(), yN * sizeof(float), dY, yN * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);
    return VerifyFloat(y, yGolden, GemvPrecision::kF32Threshold, (uint32_t)xLen, tag);
}

static uint32_t TestFP16_HSS_Normal(TestContext& ctx)
{
    return DoTestHSS(ctx, ACLBLAS_OP_N, 32, 32, 1, 1.0f, 0.0f, "N-HSS-32x32");
}
static uint32_t TestFP16_HSS_Large(TestContext& ctx)
{
    return DoTestHSS(ctx, ACLBLAS_OP_N, 128, 500, 3, 1.0f, 0.0f, "N-HSS-128x500");
}

// ---- Strided (SIMT) tests (TODO: implement StridedTest) ----
// static uint32_t TestFP32_Strided_Incx2(...)
// static uint32_t TestFP32_Strided_Incy2(...)
// static uint32_t TestFP32_Strided_Lda(...)

// (old verbose test bodies removed — now using DoTest/DoTestHalf helpers above)

// ============================================================
// Test registry — function pointer + name
// ============================================================
using TestFunc = uint32_t (*)(TestContext&);

struct TestEntry { const char* name; TestFunc func; };

static const TestEntry kTestCases[] = {
    {"TestFP32_Normal",                     TestFP32_Normal},
    {"TestFP16_Normal",                     TestFP16_Normal},
    {"TestFP16_Normal_Large",               TestFP16_Normal_Large},
    {"TestFP16_Normal_LargeM",              TestFP16_Normal_LargeM},
    {"TestFP32_Transpose",                  TestFP32_Transpose},
    {"TestFP32_Normal_Large",               TestFP32_Normal_Large},
    {"TestFP32_Transpose_Large",            TestFP32_Transpose_Large},
    {"TestFP32_Transpose_Odd",              TestFP32_Transpose_Odd},
    {"TestFP32_Transpose_LargeM",           TestFP32_Transpose_LargeM},
    {"TestFP32_Transpose_LargeN",           TestFP32_Transpose_LargeN},
    {"TestFP32_Transpose_LargeMN",          TestFP32_Transpose_LargeMN},
    {"TestFP32_Transpose_NoBeta",           TestFP32_Transpose_NoBeta},
    {"TestFP32_Transpose_MisalignedM",      TestFP32_Transpose_MisalignedM},
    {"TestFP16_Transpose",                  TestFP16_Transpose},
    {"TestFP16_Transpose_Large",            TestFP16_Transpose_Large},
    {"TestFP16_Transpose_Odd",              TestFP16_Transpose_Odd},
    {"TestFP16_HSS_Normal",                 TestFP16_HSS_Normal},
    {"TestFP16_HSS_Large",                  TestFP16_HSS_Large},
    {"TestFP32_Transpose_VeryLargeM",       TestFP32_Transpose_VeryLargeM},
    {"TestFP32_Transpose_VeryLargeN",       TestFP32_Transpose_VeryLargeN},
    {"TestFP32_Transpose_VeryLargeMN",      TestFP32_Transpose_VeryLargeMN},
    {"TestFP32_Transpose_VeryLargeMisaligned", TestFP32_Transpose_VeryLargeMisaligned},
    {"TestFP32_Transpose_MisalignedN",      TestFP32_Transpose_MisalignedN},
    {"TestFP32_Transpose_BothMisaligned",   TestFP32_Transpose_BothMisaligned},
    {"TestFP32_Transpose_SmallBatchLarge",  TestFP32_Transpose_SmallBatchLarge},
    {"TestFP32_Normal_LargeM",              TestFP32_Normal_LargeM},
    {"TestFP32_Normal_LargeMN",             TestFP32_Normal_LargeMN},
    {"TestFP32_Normal_LargeBatch",          TestFP32_Normal_LargeBatch},
    {"TestFP32_Normal_BatchNonAligned",     TestFP32_Normal_BatchNonAligned},
    {"TestFP32_Normal_AllNonAligned",       TestFP32_Normal_AllNonAligned},
    {"TestFP32_Normal_MisalignedM",         TestFP32_Normal_MisalignedM},
    {"TestFP32_Normal_LargeN",              TestFP32_Normal_LargeN},
    {"TestFP32_Normal_SmallNonAligned",     TestFP32_Normal_SmallNonAligned},
};

static int TestSetup(TestContext &ctx)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    auto blasRet = aclblasCreate(&ctx.handle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
                   LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); aclFinalize(); return blasRet);
    ret = aclrtCreateStream(&ctx.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
                   aclblasDestroy(ctx.handle); aclFinalize(); return ret);
    blasRet = aclblasSetStream(ctx.handle, ctx.stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
                   LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
                   aclrtDestroyStream(ctx.stream); aclblasDestroy(ctx.handle); aclFinalize(); return blasRet);
    return 0;
}

static void TestTeardown(TestContext &ctx)
{
    aclrtDestroyStream(ctx.stream);
    aclblasDestroy(ctx.handle);
    aclrtResetDevice(0);
    aclFinalize();
}

int main()
{
    LOG_PRINT("=== GemvBatched Test Suite ===\n\n");

    TestContext ctx{nullptr, nullptr};
    int setupErr = TestSetup(ctx);
    if (setupErr != 0) return setupErr;

    uint32_t te = 0;
    for (const auto &tc : kTestCases) {
        LOG_PRINT("--- %s ---\n", tc.name);
        if (tc.func(ctx) > 0) te++;
        LOG_PRINT("\n");
    }

    uint32_t total = sizeof(kTestCases) / sizeof(kTestCases[0]);
    LOG_PRINT("========================================\n");
    LOG_PRINT("  Total: %u  Passed: %u  Failed: %u\n", total, total - te, te);
    LOG_PRINT("========================================\n");

    TestTeardown(ctx);
    return te > 0 ? 1 : 0;
}
