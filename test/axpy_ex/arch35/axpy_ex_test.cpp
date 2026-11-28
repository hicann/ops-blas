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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "axpy_ex_param.h"
#include "axpy_ex_golden.h"
#include "axpy_ex_npu_wrapper.h"

// ── Test fixture ─────────────────────────────────────────────────────────────
class AxpyExTest : public BlasTest<AxpyExParam> {};

// ── TEST_F: null handle (TF_01, not in CSV) ─────────────────────────────────
TEST_F(AxpyExTest, NullHandle)
{
    float alphaVal = 2.5f;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM_1_1", 42);
    std::vector<float> yHost = makeBlasStrided(5, 1, "RANDOM_1_1", 42);
    aclblasStatus_t ret = aclblasAxpyEx_npu(
        nullptr, 5, &alphaVal, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, yHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    AxpyEx, AxpyExTest, ::testing::ValuesIn(GetCasesFromCsv<AxpyExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AxpyExParam>);

// ── issue #1326 负例特例：incx/incy = INT32_MIN 的 host 校验短路验证 ──────────
// 绕过 npu 包装（设备缓冲按 (n-1)*|inc|+1 分配且 std::abs 对 INT32_MIN 为 UB，
// 该步长下不可达），以最小哑设备缓冲直调公开接口；期望值写死 INVALID_VALUE，
// 保证该校验被移除时用例失败
static void RunAxpyExInt32MinCase(aclblasHandle_t handle, const AxpyExParam& p, bool incxIsMin)
{
    const size_t dummyBytes = 8 * sizeof(float);
    void* dX = nullptr;
    void* dY = nullptr;
    float alphaLocal = p.alphaVal();
    ASSERT_EQ(aclrtMalloc(&dX, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dY, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    aclblasStatus_t ret = aclblasAxpyEx(
        handle, p.n, &alphaLocal, static_cast<aclDataType>(p.alphaType), dX, static_cast<aclDataType>(p.xType),
        incxIsMin ? INT32_MIN : p.incx, dY, static_cast<aclDataType>(p.yType), incxIsMin ? p.incy : INT32_MIN,
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dX);
    aclrtFree(dY);
}

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(AxpyExTest, CsvDriven)
{
    const auto& p = GetParam();

    if (p.incx == INT32_MIN || p.incy == INT32_MIN) {
        RunAxpyExInt32MinCase(AxpyExTest::handle_, p, p.incx == INT32_MIN);
        return;
    }

    // Step 1: Generate host data (x and y independently)
    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    const void* xPtr = xHost.empty() ? nullptr : xHost.data();
    void* yPtr = yHost.empty() ? nullptr : yHost.data();

    // R01: alphaIsNull=true → alphaPtr=nullptr (triggers op alpha==nullptr check)
    float alphaLocal = p.alphaVal();
    const void* alphaPtr = p.alphaIsNull ? nullptr : &alphaLocal;

    // Step 2: Save golden copies before NPU mutates yHost
    std::vector<float> xGolden = xHost;
    std::vector<float> yGolden = yHost;

    // Step 3: Execute NPU (wrapper handles nullptrs, dtype packing, device memory)
    aclblasStatus_t ret = aclblasAxpyEx_npu(
        AxpyExTest::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xPtr,
        static_cast<aclDataType>(p.xType), p.incx, yPtr, static_cast<aclDataType>(p.yType), p.incy,
        static_cast<aclDataType>(p.executionType), p.alphaOnDevice);

    // Step 3b: Verify expected error code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n <= 0)
        return; // n=0 short-circuit, nothing to compare

    // Step 4: Compute golden (in-place on saved copy)
    aclblasStatus_t cpuRet = aclblasAxpyEx_cpu(
        AxpyExTest::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xGolden.data(),
        static_cast<aclDataType>(p.xType), p.incx, yGolden.data(), static_cast<aclDataType>(p.yType), p.incy,
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification (MIXED_TOLERANCE, threshold per yType)
    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.yType), yGolden.data(), static_cast<size_t>(std::max(0, p.n)));

    int absIncy = std::abs(p.incy);
    EXPECT_TRUE(Verifier::verifyVector(
        static_cast<const float*>(yPtr), yGolden.data(), static_cast<size_t>(std::max(0, p.n)), absIncy, cfg,
        p.caseName));
}
