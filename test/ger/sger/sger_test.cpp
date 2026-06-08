/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sger_test.cpp
 * \brief Test for SGER operation
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <random>
#include <chrono>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <gtest/gtest.h>

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

constexpr float RTOL = 1e-4f;
constexpr float ATOL = 1e-5f;

void ComputeGoldenGer(
    int m, int n, float alpha, const std::vector<float>& x, int incx, const std::vector<float>& y,
    int incy, std::vector<float>& A, int lda)
{
    for (int i = 0; i < m; i++) {
        float xi = x[i * incx];
        for (int j = 0; j < n; j++) {
            float yj = y[j * incy];
            A[i * lda + j] += alpha * xi * yj;
        }
    }
}

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden, int m, int n, int lda)
{
    auto printTensor = [](std::vector<float>& tensor, int rows, int cols, int lda, const char* name) {
        std::cout << name << " (m=" << rows << ", n=" << cols << ", lda=" << lda << "):" << std::endl;
        for (int i = 0; i < std::min(rows, int(4)); i++) {
            std::cout << "  row " << i << ": ";
            for (int j = 0; j < std::min(cols, int(8)); j++) {
                std::cout << tensor[i * lda + j] << " ";
            }
            if (cols > 8)
                std::cout << "...";
            std::cout << std::endl;
        }
        if (rows > 4)
            std::cout << "  ..." << std::endl;
    };

    printTensor(output, m, n, lda, "Output");
    printTensor(golden, m, n, lda, "Golden");

    bool allMatch = true;
    size_t maxErrors = 10;
    size_t errorCount = 0;

    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::fabs(output[i] - golden[i]);
        float relDiff = (golden[i] != 0.0f) ? diff / std::fabs(golden[i]) : diff;

        if (diff > ATOL && relDiff > RTOL) {
            if (errorCount < maxErrors) {
                std::cout << "Error at index " << i << ": output=" << output[i] << ", golden=" << golden[i]
                          << ", diff=" << diff << std::endl;
            }
            errorCount++;
            allMatch = false;
        }
    }

    if (allMatch) {
        std::cout << "[Success] Case accuracy verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy verification failed! " << errorCount << " errors found." << std::endl;
        return 1;
    }
}

TEST(SgerTest, Basic)
{
    int32_t deviceId = 0;

    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    constexpr int m = 4096;
    constexpr int n = 2048;
    constexpr int lda = 4096;
    constexpr float alpha = 2.0f;
    constexpr int incx = 1;
    constexpr int incy = 1;

    std::vector<float> A(m * lda, 0.0f);
    std::vector<float> x(m, 0.0f);
    std::vector<float> y(n, 0.0f);

    for (int i = 0; i < m; i++) {
        x[i] = dis(gen);
    }

    for (int j = 0; j < n; j++) {
        y[j] = dis(gen);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * lda + j] = dis(gen);
        }
    }

    std::vector<float> golden = A;

    ComputeGoldenGer(m, n, alpha, x, incx, y, incy, golden, lda);

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    aclblasStatus_t ret = aclblasSger(handle, m, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    ASSERT_EQ(VerifyResult(A, golden, m, n, lda), 0u);
}
