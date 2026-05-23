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

constexpr float EPSILON = 1e-2f;

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
        float relDiff = diff / (std::abs(golden[i]) + 1e-6f);
        if (relDiff > EPSILON && diff > EPSILON) {
            std::cout << "[Failed] Index " << i << ": output=" << output[i] << " golden=" << golden[i] << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

void ComputeGoldenStrmmRowMajor(const float *A, const float *B, float *C, int64_t m, int64_t n, int64_t k,
                                int64_t lda, int64_t ldb, int64_t ldc, float alpha, int64_t transa, int64_t transb)
{
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                float a_val, b_val;
                
                if (transa == 0) {
                    a_val = A[i * lda + l];
                } else {
                    a_val = A[l * lda + i];
                }
                
                if (transb == 0) {
                    b_val = B[l * ldb + j];
                } else {
                    b_val = B[j * ldb + l];
                }
                
                sum += a_val * b_val;
            }
            C[i * ldc + j] = alpha * sum;
        }
    }
}

int TestStrmmBasic()
{
    std::cout << "Testing strmm (side=L, uplo=L, transa=N, transb=N, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 128;
    constexpr int64_t N = 128;
    constexpr int64_t K = 128;
    constexpr int64_t lda = K;
    constexpr int64_t ldb = N;
    constexpr int64_t ldc = N;
    constexpr float alpha = 1.0f;

    std::vector<float> A(M * K, 0.0f);
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j <= i; j++) {
            A[i * lda + j] = 1.0f;
        }
    }
    
    std::vector<float> B(K * N, 1.0f);
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < N; j++) {
            B[i * ldb + j] = 1.0f;
        }
    }
    
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> CGolden(M * N, 0.0f);

    ComputeGoldenStrmmRowMajor(A.data(), B.data(), CGolden.data(), M, N, K, lda, ldb, ldc, alpha, 0, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    uint8_t *aDevice = nullptr;
    uint8_t *bDevice = nullptr;
    uint8_t *cDevice = nullptr;
    size_t aByteSize = M * K * sizeof(float);
    size_t bByteSize = K * N * sizeof(float);
    size_t cByteSize = M * N * sizeof(float);
    aclError aclRet = aclrtMalloc((void **)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&bDevice, bByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&cDevice, cByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevice, bByteSize, B.data(), bByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevice, cByteSize, C.data(), cByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmm(handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                       M, N, alpha, aDevice, lda, bDevice, ldb, cDevice, ldc);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmm failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(C.data(), cByteSize, cDevice, cByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy C failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(C, CGolden);
}

int TestStrmmTransA()
{
    std::cout << "Testing strmm (side=L, uplo=L, transa=T, transb=N, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 128;
    constexpr int64_t N = 128;
    constexpr int64_t K = 128;
    constexpr int64_t lda = M;
    constexpr int64_t ldb = N;
    constexpr int64_t ldc = N;
    constexpr float alpha = 2.0f;

    std::vector<float> A(K * M, 0.0f);
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = i; j < M; j++) {
            A[i * lda + j] = 1.0f;
        }
    }
    
    std::vector<float> B(K * N, 1.0f);
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < N; j++) {
            B[i * ldb + j] = 1.0f;
        }
    }
    
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> CGolden(M * N, 0.0f);

    ComputeGoldenStrmmRowMajor(A.data(), B.data(), CGolden.data(), M, N, K, lda, ldb, ldc, alpha, 1, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    uint8_t *aDevice = nullptr;
    uint8_t *bDevice = nullptr;
    uint8_t *cDevice = nullptr;
    size_t aByteSize = K * M * sizeof(float);
    size_t bByteSize = K * N * sizeof(float);
    size_t cByteSize = M * N * sizeof(float);
    aclError aclRet = aclrtMalloc((void **)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&bDevice, bByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&cDevice, cByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevice, bByteSize, B.data(), bByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevice, cByteSize, C.data(), cByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmm(handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT,
                       M, N, alpha, aDevice, lda, bDevice, ldb, cDevice, ldc);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmm failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(C.data(), cByteSize, cDevice, cByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy C failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(C, CGolden);
}

int TestStrmmTransB()
{
    std::cout << "Testing strmm (side=L, uplo=L, transa=N, transb=T, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 128;
    constexpr int64_t N = 128;
    constexpr int64_t K = 128;
    constexpr int64_t lda = K;
    constexpr int64_t ldb = K;
    constexpr int64_t ldc = N;
    constexpr float alpha = 0.5f;

    std::vector<float> A(M * K, 0.0f);
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j <= i; j++) {
            A[i * lda + j] = 1.0f;
        }
    }
    
    std::vector<float> B(N * K, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < K; j++) {
            B[i * ldb + j] = 1.0f;
        }
    }
    
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> CGolden(M * N, 0.0f);

    ComputeGoldenStrmmRowMajor(A.data(), B.data(), CGolden.data(), M, N, K, lda, ldb, ldc, alpha, 0, 1);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    uint8_t *aDevice = nullptr;
    uint8_t *bDevice = nullptr;
    uint8_t *cDevice = nullptr;
    size_t aByteSize = M * K * sizeof(float);
    size_t bByteSize = N * K * sizeof(float);
    size_t cByteSize = M * N * sizeof(float);
    aclError aclRet = aclrtMalloc((void **)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&bDevice, bByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&cDevice, cByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevice, bByteSize, B.data(), bByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy bDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevice, cByteSize, C.data(), cByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmm(handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                       M, N, alpha, aDevice, lda, bDevice, ldb, cDevice, ldc);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmm failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(C.data(), cByteSize, cDevice, cByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy C failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(C, CGolden);
}

int32_t main(int32_t argc, char *argv[])
{
    int ret1 = TestStrmmBasic();
    int ret2 = TestStrmmTransA();
    int ret3 = TestStrmmTransB();

    if (ret1 == 0 && ret2 == 0 && ret3 == 0) {
        std::cout << "[PASS] strmm_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] strmm_test" << std::endl;
        return 1;
    }
}