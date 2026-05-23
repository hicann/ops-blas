/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
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

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    constexpr std::complex<float> alpha(1.0f, 0.0f);
    constexpr std::complex<float> beta(0.0f, 0.0f);
    constexpr int64_t lda = K;
    constexpr int64_t ldb = N;
    constexpr int64_t ldc = N;

    uint8_t *aDevice = nullptr;
    uint8_t *bDevice = nullptr;
    uint8_t *cDevice = nullptr;
    size_t aByteSize = 2 * batchCount * M * K * sizeof(float);
    size_t bByteSize = 2 * batchCount * K * N * sizeof(float);
    size_t cByteSize = 2 * batchCount * M * N * sizeof(float);
    aclError aclRet = aclrtMalloc((void **)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&bDevice, bByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&cDevice, cByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, AFlat.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevice, bByteSize, BFlat.data(), bByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevice, cByteSize, CFlat.data(), cByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgemmBatched(handle, ACLBLAS_OP_N, ACLBLAS_OP_N, M, N, K, alpha, aDevice, lda, bDevice, ldb, beta, cDevice, ldc, batchCount);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemmBatched failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(CFlat.data(), cByteSize, cDevice, cByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy CFlat failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
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