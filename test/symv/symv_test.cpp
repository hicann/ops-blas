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
* \file symv_test.cpp
* \brief
*/

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

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

enum class SymvCompareMode {
    FullMatrix,
    LowerTriangleOnly,
    SymmetricOnly,
};

constexpr SymvCompareMode kCompareMode = SymvCompareMode::FullMatrix;

static const char *GetCompareModeName()
{
    switch (kCompareMode) {
        case SymvCompareMode::FullMatrix:
            return "full-matrix";
        case SymvCompareMode::LowerTriangleOnly:
            return "lower-triangle-only";
        case SymvCompareMode::SymmetricOnly:
            return "symmetric-only";
        default:
            return "unknown";
    }
}

static std::vector<float> BuildGolden(const std::vector<float> &a, const std::vector<float> &x,
    const std::vector<float> &y, uint32_t n, uint32_t lda, int64_t incx, int64_t incy, float alpha, float beta)
{
    auto matrixIndex = [lda](uint32_t row, uint32_t col) {
        return static_cast<size_t>(row) * lda + col;
    };

    std::vector<float> golden(n, 0.0f);
    switch (kCompareMode) {
        case SymvCompareMode::FullMatrix:
            for (uint32_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t row = i >= j ? i : j;
                    uint32_t col = i >= j ? j : i;
                    acc += a[matrixIndex(row, col)] * x[j * incx];
                }
                golden[i] = alpha * acc + beta * y[i * incy];
            }
            break;
        case SymvCompareMode::LowerTriangleOnly:
            for (uint32_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    acc += a[matrixIndex(i, j)] * x[j * incx];
                }
                golden[i] = alpha * acc + beta * y[i * incy];
            }
            break;
        case SymvCompareMode::SymmetricOnly:
            for (uint32_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (uint32_t i = j + 1; i < n; ++i) {
                    acc += a[matrixIndex(i, j)] * x[i * incx];
                }
                golden[j] = alpha * acc + beta * y[j * incy];
            }
            break;
        default:
            break;
    }

    return golden;
}

static uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    std::cout << std::fixed << std::setprecision(6);

    auto printTensor = [](std::vector<float> &tensor, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    constexpr float absTol = 1e-3f;
    constexpr float relTol = 1e-4f;
    auto closeEnough = [&](float a, float b) {
        float diff = std::abs(a - b);
        float scale = std::max(std::abs(a), std::abs(b));
        return diff <= absTol || diff <= relTol * scale;
    };

    for (size_t i = 0; i < output.size(); ++i) {
        if (!closeEnough(output[i], golden[i])) {
            std::cout << "[Failed] Case accuracy is verification failed at index " << i << " (" << output[i]
                      << " vs " << golden[i] << ")" << std::endl;
            return 1;
        }
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static int RunCase(uint32_t n, uint32_t lda)
{
    int32_t deviceId = 0;
    constexpr float alpha = 0.8f;
    constexpr float beta = 1.2f;
    const int64_t incx = 1;
    const int64_t incy = 1;

    std::vector<float> a(static_cast<size_t>(n) * lda, 0.0f);
    std::vector<float> x(n, 0.0f);
    std::vector<float> y(n, 0.0f);
    std::mt19937 rng(20260324U + n + lda);
    std::uniform_real_distribution<float> dist(0.0f, 0.5f);
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
        for (uint32_t j = 0; j <= i; ++j) {
            a[static_cast<size_t>(i) * lda + j] = dist(rng);
        }
    }

    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    std::vector<float> z(n, 0.0f);
    int ret = aclblasSymv(a.data(), lda, x.data(), y.data(), z.data(), alpha, beta, n, incx, incy, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSymv failed. ERROR: %d\n", ret); return ret);

    std::vector<float> golden = BuildGolden(a, x, y, n, lda, incx, incy, alpha, beta);
    int status = VerifyResult(z, golden);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return status;
}

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::cout << "Verification mode: " << GetCompareModeName() << std::endl;
    int ret = RunCase(6144, 6144);
    if (ret != 0) {
        return ret;
    }
    return RunCase(5000, 6000);
}