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

void ComputeGoldenGer(int64_t m, int64_t n, float alpha,
                      const std::vector<float>& x, int64_t incx,
                      const std::vector<float>& y, int64_t incy,
                      std::vector<float>& A, int64_t lda)
{
    for (int64_t i = 0; i < m; i++) {
        float xi = x[i * incx];
        for (int64_t j = 0; j < n; j++) {
            float yj = y[j * incy];
            A[i * lda + j] += alpha * xi * yj;
        }
    }
}

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden, int64_t m, int64_t n, int64_t lda)
{
    auto printTensor = [](std::vector<float>& tensor, int64_t rows, int64_t cols, int64_t lda, const char* name) {
        std::cout << name << " (m=" << rows << ", n=" << cols << ", lda=" << lda << "):" << std::endl;
        for (int64_t i = 0; i < std::min(rows, int64_t(4)); i++) {
            std::cout << "  row " << i << ": ";
            for (int64_t j = 0; j < std::min(cols, int64_t(8)); j++) {
                std::cout << tensor[i * lda + j] << " ";
            }
            if (cols > 8) std::cout << "...";
            std::cout << std::endl;
        }
        if (rows > 4) std::cout << "  ..." << std::endl;
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
                std::cout << "Error at index " << i << ": output=" << output[i]
                          << ", golden=" << golden[i]
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

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    uint32_t seed;
    if (argc > 1) {
        seed = static_cast<uint32_t>(std::atoi(argv[1]));
        std::cout << "Using seed from command line: " << seed << std::endl;
    } else {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::cout << "Using time-based seed: " << seed << std::endl;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);


    constexpr int64_t m = 4096;
    constexpr int64_t n = 2048;
    constexpr int64_t lda = 4096;
    constexpr float alpha = 2.0f;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;

    std::cout << "Sger Test (m=" << m << ", n=" << n << ", lda=" << lda << ")" << std::endl;

    std::vector<float> A(m * lda, 0.0f);
    std::vector<float> x(m, 0.0f);
    std::vector<float> y(n, 0.0f);


    for (int64_t i = 0; i < m; i++) {
        x[i] = dis(gen);
    }

    for (int64_t j = 0; j < n; j++) {
        y[j] = dis(gen);
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A[i * lda + j] = dis(gen);
        }
    }

    std::vector<float> golden = A;

    ComputeGoldenGer(m, n, alpha, x, incx, y, incy, golden, lda);

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasSger(nullptr, m, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSger failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(A, golden, m, n, lda);
}
