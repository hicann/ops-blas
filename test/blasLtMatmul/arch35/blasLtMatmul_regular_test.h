/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "cann_ops_blasLt.h"
#include "device.h"
#include "dtype_cast.h"

namespace lt_regular_test {

struct Resources {
    aclblasLtMatmulDesc_t desc = nullptr;
    aclblasLtMatrixLayout_t layouts[4]{};
    ~Resources()
    {
        for (auto layout : layouts) {
            if (layout != nullptr) {
                aclblasLtMatrixLayoutDestroy(layout);
            }
        }
        if (desc != nullptr) {
            aclblasLtMatmulDescDestroy(desc);
        }
    }
};

inline std::vector<uint8_t> Encode(const std::vector<float>& values, aclDataType type)
{
    const uint8_t e4m3[] = {0xc0, 0xb8, 0xb0, 0x30, 0x38, 0x3c, 0x40, 0x44};
    const uint8_t e5m2[] = {0xc0, 0xbc, 0xb8, 0x38, 0x3c, 0x3e, 0x40, 0x42};
    const float fp8Values[] = {-2, -1, -0.5f, 0.5f, 1, 1.5f, 2, 3};
    bool fp8 = type == ACL_FLOAT8_E4M3FN || type == ACL_FLOAT8_E5M2;
    std::vector<uint8_t> result(values.size() * (fp8 ? 1 : 2));
    for (size_t i = 0; i < values.size(); ++i) {
        if (fp8) {
            size_t index = static_cast<size_t>(std::find(fp8Values, fp8Values + 8, values[i]) - fp8Values);
            result[i] = type == ACL_FLOAT8_E4M3FN ? e4m3[index] : e5m2[index];
        } else {
            uint16_t bits = type == ACL_BF16 ? blas_common::FloatToBf16(values[i]) :
                blas_common::FloatToHalf(values[i]);
            result[2 * i] = static_cast<uint8_t>(bits);
            result[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
        }
    }
    return result;
}

inline void Init(Resources& r, aclDataType a, aclDataType b, aclDataType output,
    bool transA, bool transB, int m, int n, int k, const int ld[4])
{
    bool integer = a == ACL_INT8;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&r.desc, integer ? ACLBLAS_COMPUTE_32I : ACLBLAS_COMPUTE_32F,
        integer ? ACL_INT32 : ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const aclblasOperation_t operations[] = {
        transA ? ACLBLAS_OP_T : ACLBLAS_OP_N,
        transB ? ACLBLAS_OP_T : ACLBLAS_OP_N
    };
    ASSERT_EQ(aclblasLtMatmulDescSetAttribute(r.desc, ACLBLASLT_MATMUL_DESC_TRANSA,
        &operations[0], sizeof(operations[0])), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatmulDescSetAttribute(r.desc, ACLBLASLT_MATMUL_DESC_TRANSB,
        &operations[1], sizeof(operations[1])), ACLBLAS_STATUS_SUCCESS);
    const aclDataType types[] = {a, b, output, output};
    const int rows[] = {transA ? k : m, transB ? n : k, m, m};
    const int cols[] = {transA ? m : k, transB ? k : n, n, n};
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(aclblasLtMatrixLayoutCreate(&r.layouts[i], types[i], rows[i], cols[i], ld[i]),
            ACLBLAS_STATUS_SUCCESS);
        const auto order = ACLBLASLT_ORDER_ROW;
        ASSERT_EQ(aclblasLtMatrixLayoutSetAttribute(r.layouts[i], ACLBLASLT_MATRIX_LAYOUT_ORDER,
            &order, sizeof(order)), ACLBLAS_STATUS_SUCCESS);
    }
}

inline void CheckQueries(aclblasLtHandle_t handle, Resources& r, aclDataType a, aclDataType b,
    aclDataType output, size_t workspace, aclblasLtMatmulAlgo_t& algo)
{
    bool integer = a == ACL_INT8;
    int ids[4]{};
    int count = 0;
    auto compute = integer ? ACLBLAS_COMPUTE_32I : ACLBLAS_COMPUTE_32F;
    auto scalar = integer ? ACL_INT32 : ACL_FLOAT;
    ASSERT_EQ(aclblasLtMatmulAlgoGetIds(handle, compute, scalar, a, b, output, output, 4, ids, &count),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(aclblasLtMatmulAlgoInit(handle, compute, scalar, a, b, output, output, ids[0], &algo),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(algo.max_workspace_bytes, 0U);
    EXPECT_EQ(aclblasLtMatmulAlgoInit(handle, compute, scalar, a, b, output, output, ids[0] + 1, &algo),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    aclblasLtMatmulHeuristicResult_t results[3]{};
    ASSERT_EQ(aclblasLtMatmulAlgoGetHeuristic(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2],
        r.layouts[3], nullptr, 3, results, &count), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(results[0].workspaceSize, workspace);
    algo = results[0].algo;
    aclblasLtMatmulHeuristicResult_t checked{};
    ASSERT_EQ(aclblasLtMatmulAlgoCheck(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2], r.layouts[3],
        &algo, &checked), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(checked.workspaceSize, workspace);
    auto modified = algo;
    const uint32_t stages = 2;
    ASSERT_EQ(aclblasLtMatmulAlgoConfigSetAttribute(&modified, ACLBLASLT_ALGO_CONFIG_STAGES_ID,
        &stages, sizeof(stages)), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasLtMatmulAlgoCheck(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2], r.layouts[3],
        &modified, &checked), ACLBLAS_STATUS_NOT_SUPPORTED);
    aclblasLtMatmulPreferenceOpaque_t preference{};
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&preference), ACLBLAS_STATUS_SUCCESS);
    const size_t insufficient = workspace - 1;
    ASSERT_EQ(aclblasLtMatmulPreferenceSetAttribute(&preference, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &insufficient, sizeof(insufficient)), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasLtMatmulAlgoGetHeuristic(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2],
        r.layouts[3], &preference, 3, results, &count), ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(count, 0);
}

inline void VerifyFloatOutput(
    const std::vector<uint16_t>& actual, const std::vector<float>& a, const std::vector<float>& b,
    int m, int n, int k, const int ld[4], aclDataType output, bool transA, bool transB, bool alias,
    float alpha, float beta)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < ld[3]; ++col) {
            float expected = alias ? 1.5f : -2.0f;
            if (col < n) {
                float sum = 0;
                for (int inner = 0; inner < k; ++inner) {
                    sum += a[transA ? inner * ld[0] + row : row * ld[0] + inner] *
                           b[transB ? col * ld[1] + inner : inner * ld[1] + col];
                }
                expected = alpha * sum + beta * 1.5f;
            }
            float result = output == ACL_BF16 ? blas_common::Bf16ToFloat(actual[row * ld[3] + col]) :
                blas_common::HalfToFloat(actual[row * ld[3] + col]);
            float tolerance = col < n ? (output == ACL_BF16 ? 0.008f : 0.001f) *
                std::max(1.0f, std::abs(expected)) : 0.0f;
            EXPECT_NEAR(result, expected, tolerance) << "row=" << row << " col=" << col;
        }
    }
}

inline void RunFloat(aclblasLtHandle_t handle, aclrtStream stream, aclDataType aType, aclDataType bType,
    aclDataType output, bool transA, bool transB, bool alias, bool direct = false)
{
    constexpr int m = 19;
    constexpr int n = 17;
    constexpr int k = 35;
    const int ld[] = {(transA ? m : k) + 3, (transB ? k : n) + 5, n + 7, n + (alias ? 7 : 9)};
    Resources r;
    Init(r, aType, bType, output, transA, transB, m, n, k, ld);
    const float values[] = {-2, -1, -0.5f, 0.5f, 1, 1.5f, 2, 3};
    std::vector<float> a((transA ? k : m) * ld[0]);
    std::vector<float> b((transB ? n : k) * ld[1]);
    std::vector<float> c(m * ld[2], 1.5f);
    std::vector<float> initialD(m * ld[3], -2.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = values[(i * 3 + 1) % 8];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = values[(i * 5 + 3) % 8];
    }
    auto aRaw = Encode(a, aType);
    auto bRaw = Encode(b, bType);
    auto cRaw = Encode(c, output);
    auto dRaw = Encode(initialD, output);
    DeviceBuffer dA(aRaw.size());
    DeviceBuffer dB(bRaw.size());
    DeviceBuffer dC(cRaw.size());
    DeviceBuffer dD(dRaw.size());
    DeviceBuffer workspace(m * n * 4);
    dA.copyFromHost(aRaw.data(), aRaw.size());
    dB.copyFromHost(bRaw.data(), bRaw.size());
    dC.copyFromHost(cRaw.data(), cRaw.size());
    dD.copyFromHost(dRaw.data(), dRaw.size());
    aclblasLtMatmulAlgo_t algo{};
    CheckQueries(handle, r, aType, bType, output, workspace.size(), algo);
    const float alpha = direct ? 1.0f : 1.25f;
    const float beta = direct ? 0.0f : -0.5f;
    ASSERT_EQ(aclblasLtMatmul(handle, r.desc, &alpha, dA.ptr(), r.layouts[0], dB.ptr(), r.layouts[1], &beta,
        dC.ptr(), r.layouts[2], alias ? dC.ptr() : dD.ptr(), r.layouts[3], direct ? nullptr : &algo,
        direct ? nullptr : workspace.ptr(), direct ? 0 : workspace.size(), stream), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    std::vector<uint16_t> actual(m * ld[3]);
    (alias ? dC : dD).copyToHost(actual.data(), actual.size() * 2);
    VerifyFloatOutput(actual, a, b, m, n, k, ld, output, transA, transB, alias, alpha, beta);
}

inline void VerifyIntegerOutput(const std::vector<int32_t>& actual, const std::vector<int8_t>& a,
    const std::vector<int8_t>& b, const std::vector<int32_t>& c, int m, int n, int k, const int ld[4],
    bool transA, bool transB, bool alias, int32_t alpha, int32_t beta)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < ld[3]; ++col) {
            int64_t expected = alias ? c[row * ld[2] + col] : -17;
            if (col < n) {
                int64_t product = 0;
                for (int inner = 0; inner < k; ++inner) {
                    product += a[transA ? inner * ld[0] + row : row * ld[0] + inner] *
                               b[transB ? col * ld[1] + inner : inner * ld[1] + col];
                }
                expected = alpha * product + beta * static_cast<int64_t>(c[row * ld[2] + col]);
            }
            EXPECT_EQ(actual[row * ld[3] + col], expected) << "row=" << row << " col=" << col;
        }
    }
}

inline void RunInteger(aclblasLtHandle_t handle, aclrtStream stream, bool transA, bool transB, bool alias)
{
    constexpr int m = 3;
    constexpr int n = 5;
    constexpr int k = 7;
    const int ld[] = {transA ? 4 : 8, 8, 8, alias ? 8 : 12};
    Resources r;
    Init(r, ACL_INT8, ACL_INT8, ACL_INT32, transA, transB, m, n, k, ld);
    std::vector<int8_t> a((transA ? k : m) * ld[0]);
    std::vector<int8_t> b((transB ? n : k) * ld[1]);
    std::vector<int32_t> c(m * ld[2]);
    std::vector<int32_t> initialD(m * ld[3], -17);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int8_t>(i % 3 + 1);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<int8_t>((i * 5) % 3 + 1);
    }
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = 16777217 + static_cast<int32_t>(i % 3);
    }
    DeviceBuffer dA(a.size());
    DeviceBuffer dB(b.size());
    DeviceBuffer dC(c.size() * 4);
    DeviceBuffer dD(initialD.size() * 4);
    DeviceBuffer workspace(m * n * 4);
    dA.copyFromHost(a.data(), a.size());
    dB.copyFromHost(b.data(), b.size());
    dC.copyFromHost(c.data(), c.size() * 4);
    dD.copyFromHost(initialD.data(), initialD.size() * 4);
    aclblasLtMatmulAlgo_t algo{};
    CheckQueries(handle, r, ACL_INT8, ACL_INT8, ACL_INT32, workspace.size(), algo);
    const int32_t alpha = 16777217;
    const int32_t beta = 1;
    ASSERT_EQ(aclblasLtMatmul(handle, r.desc, &alpha, dA.ptr(), r.layouts[0], dB.ptr(), r.layouts[1], &beta,
        dC.ptr(), r.layouts[2], alias ? dC.ptr() : dD.ptr(), r.layouts[3], &algo, workspace.ptr(),
        workspace.size(), stream), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    std::vector<int32_t> actual(m * ld[3]);
    (alias ? dC : dD).copyToHost(actual.data(), actual.size() * 4);
    VerifyIntegerOutput(actual, a, b, c, m, n, k, ld, transA, transB, alias, alpha, beta);
}

inline void ExpectRejected(aclblasLtHandle_t handle, aclrtStream stream, Resources& r,
    const aclblasLtMatmulAlgo_t& algo, aclblasStatus_t expected)
{
    aclblasLtMatmulHeuristicResult_t result{};
    int count = 0;
    EXPECT_EQ(aclblasLtMatmulAlgoGetHeuristic(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2],
        r.layouts[3], nullptr, 1, &result, &count), expected);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(aclblasLtMatmulAlgoCheck(handle, r.desc, r.layouts[0], r.layouts[1], r.layouts[2], r.layouts[3],
        &algo, &result), expected);
    float alpha = 1;
    float beta = 0;
    alignas(16) float dummy[4]{};
    EXPECT_EQ(aclblasLtMatmul(handle, r.desc, &alpha, dummy, r.layouts[0], dummy, r.layouts[1], &beta,
        dummy, r.layouts[2], dummy, r.layouts[3], &algo, nullptr, 0, stream), expected);
}

inline void RunRejections(aclblasLtHandle_t handle, aclrtStream stream)
{
    Resources r;
    const int ld[] = {8, 8, 8, 8};
    Init(r, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT16, false, false, 3, 5, 7, ld);
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_EQ(aclblasLtMatmulAlgoInit(handle, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FLOAT8_E5M2, ACL_FLOAT16, ACL_FLOAT16, 0, &algo), ACLBLAS_STATUS_SUCCESS);
    int32_t mode = ACLBLASLT_POINTER_MODE_DEVICE;
    ASSERT_EQ(aclblasLtMatmulDescSetAttribute(r.desc, ACLBLASLT_MATMUL_DESC_POINTER_MODE,
        &mode, sizeof(mode)), ACLBLAS_STATUS_SUCCESS);
    ExpectRejected(handle, stream, r, algo, ACLBLAS_STATUS_NOT_SUPPORTED);
    mode = ACLBLASLT_POINTER_MODE_HOST;
    ASSERT_EQ(aclblasLtMatmulDescSetAttribute(r.desc, ACLBLASLT_MATMUL_DESC_POINTER_MODE,
        &mode, sizeof(mode)), ACLBLAS_STATUS_SUCCESS);
    float scale = 1;
    const void* scalePointer = &scale;
    ASSERT_EQ(aclblasLtMatmulDescSetAttribute(r.desc, ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &scalePointer, sizeof(void*)), ACLBLAS_STATUS_SUCCESS);
    ExpectRejected(handle, stream, r, algo, ACLBLAS_STATUS_NOT_SUPPORTED);

    Resources emptyK;
    Init(emptyK, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, false, false, 3, 5, 0, ld);
    ExpectRejected(handle, stream, emptyK, algo, ACLBLAS_STATUS_INVALID_VALUE);
    Resources largeLd;
    Init(largeLd, ACL_BF16, ACL_BF16, ACL_BF16, false, false, 3, 5, 7, ld);
    const int64_t oversized = 2147483648LL;
    ASSERT_EQ(aclblasLtMatrixLayoutSetAttribute(largeLd.layouts[0], ACLBLASLT_MATRIX_LAYOUT_LD,
        &oversized, sizeof(oversized)), ACLBLAS_STATUS_SUCCESS);
    ExpectRejected(handle, stream, largeLd, algo, ACLBLAS_STATUS_NOT_SUPPORTED);
}

} // namespace lt_regular_test
