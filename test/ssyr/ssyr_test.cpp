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
#include <iostream>
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

constexpr float EPSILON = 1e-3f;

uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    constexpr size_t maxPrintSize = 10;
    std::cout << "Output: ";
    for (size_t i = 0; i < std::min(output.size(), maxPrintSize); i++) {
        std::cout << output[i] << " ";
    }
    if (output.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    std::cout << "Golden: ";
    for (size_t i = 0; i < std::min(golden.size(), maxPrintSize); i++) {
        std::cout << golden[i] << " ";
    }
    if (golden.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output[i] - golden[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Index " << i << ": output=" << output[i] << " golden=" << golden[i] << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

void ComputeGoldenSsyrUpper(float *A, const float *x, float alpha, int64_t n, int64_t lda)
{
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

void ComputeGoldenSsyrLower(float *A, const float *x, float alpha, int64_t n, int64_t lda)
{
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

int TestSsyrUpper()
{
    std::cout << "Testing ssyr (uplo=U):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;
    constexpr float alpha = 2.0f;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < i; j++) {
            A[i * lda + j] = 0.0f;
        }
        A[i * lda + i] = 3.0f;
    }
    std::vector<float> x(N);
    for (int64_t i = 0; i < N; i++) {
        x[i] = i + 1;
    }
    std::vector<float> AGolden = A;

    ComputeGoldenSsyrUpper(AGolden.data(), x.data(), alpha, N, lda);

    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
    handle = stream;

    auto ret = aclblasSsyr(handle, ACLBLAS_UPPER, N, alpha, x.data(), incx, A.data(), lda, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyr failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(A, AGolden);
}

int TestSsyrLower()
{
    std::cout << "Testing ssyr (uplo=L):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;
    constexpr float alpha = 2.0f;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = i + 1; j < N; j++) {
            A[i * lda + j] = 0.0f;
        }
        A[i * lda + i] = 3.0f;
    }
    std::vector<float> x(N);
    for (int64_t i = 0; i < N; i++) {
        x[i] = i + 1;
    }
    std::vector<float> AGolden = A;

    ComputeGoldenSsyrLower(AGolden.data(), x.data(), alpha, N, lda);

    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
    handle = stream;

    auto ret = aclblasSsyr(handle, ACLBLAS_LOWER, N, alpha, x.data(), incx, A.data(), lda, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyr failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(A, AGolden);
}

int32_t main(int32_t argc, char *argv[])
{
    int ret1 = TestSsyrUpper();
    int ret2 = TestSsyrLower();

    if (ret1 == 0 && ret2 == 0) {
        std::cout << "[PASS] ssyr_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] ssyr_test" << std::endl;
        return 1;
    }
}