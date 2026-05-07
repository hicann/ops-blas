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
#include <complex>
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
    constexpr size_t maxPrintSize = 20;
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

void ComputeGoldenCgemmBatched(std::vector<std::complex<float>> &A,
                                std::vector<std::complex<float>> &B,
                                std::vector<std::complex<float>> &C,
                                int64_t m, int64_t k, int64_t n, int64_t batch)
{
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                std::complex<float> sum(0.0f, 0.0f);
                for (int64_t l = 0; l < k; l++) {
                    sum += A[b * m * k + i * k + l] * B[b * k * n + l * n + j];
                }
                C[b * m * n + i * n + j] += sum;
            }
        }
    }
}

int TestCgemmBatched()
{
    std::cout << "Testing cgemm_batched:" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 16;
    constexpr int64_t K = 16;
    constexpr int64_t N = 16;
    constexpr int64_t batchCount = 4;

    std::vector<std::complex<float>> A(batchCount * M * K);
    std::vector<std::complex<float>> B(batchCount * K * N);
    std::vector<std::complex<float>> C(batchCount * M * N, std::complex<float>(0.0f, 0.0f));

    for (int64_t b = 0; b < batchCount; b++) {
        for (int64_t i = 0; i < M * K; i++) {
            A[b * M * K + i] = std::complex<float>(i + 1, i + 1);
        }
        for (int64_t i = 0; i < K * N; i++) {
            B[b * K * N + i] = std::complex<float>(i + 1, -i - 1);
        }
    }

    std::vector<std::complex<float>> CGolden = C;
    ComputeGoldenCgemmBatched(A, B, CGolden, M, K, N, batchCount);

    std::vector<float> AFlat(2 * batchCount * M * K);
    std::vector<float> BFlat(2 * batchCount * K * N);
    std::vector<float> CFlat(2 * batchCount * M * N, 0.0f);
    std::vector<float> CGoldenFlat(2 * batchCount * M * N);

    for (size_t i = 0; i < A.size(); i++) {
        AFlat[2 * i] = A[i].real();
        AFlat[2 * i + 1] = A[i].imag();
    }
    for (size_t i = 0; i < B.size(); i++) {
        BFlat[2 * i] = B[i].real();
        BFlat[2 * i + 1] = B[i].imag();
    }
    for (size_t i = 0; i < CGolden.size(); i++) {
        CGoldenFlat[2 * i] = CGolden[i].real();
        CGoldenFlat[2 * i + 1] = CGolden[i].imag();
    }

    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
    handle = stream;

    auto ret = aclblasCgemmBatched(handle, M, K, N, batchCount, AFlat.data(), K, BFlat.data(), N, CFlat.data(), N, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemmBatched failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(CFlat, CGoldenFlat);
}

int32_t main(int32_t argc, char *argv[])
{
    int ret = TestCgemmBatched();

    if (ret == 0) {
        std::cout << "[PASS] cgemm_batched_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] cgemm_batched_test" << std::endl;
        return 1;
    }
}