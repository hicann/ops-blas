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
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"

constexpr float EPSILON = 1e-3f;

static uint32_t VerifyResult(const std::vector<float> &output, const std::vector<float> &golden)
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
            std::cout << "[Failed] Index " << i << ": output=" << output[i]
                      << " golden=" << golden[i] << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static void ComputeGoldenSsyrUpper(float *A, const float *x, float alpha, int n, int lda)
{
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

static void ComputeGoldenSsyrLower(float *A, const float *x, float alpha, int n, int lda)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

static int TestSsyrUpper()
{
    std::cout << "Testing syr A2 (uplo=U):" << std::endl;

    int32_t deviceId = 0;
    constexpr int N = 128;
    constexpr int lda = N;
    constexpr int incx = 1;
    constexpr float alphaVal = 2.0f;

    std::vector<float> A(N * lda, 1.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i * lda + j] = 0.0f;
        }
        A[i * lda + i] = 3.0f;
    }
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i + 1);
    }
    std::vector<float> AGolden = A;

    ComputeGoldenSsyrUpper(AGolden.data(), x.data(), alphaVal, N, lda);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasCreate failed. ERROR: " << ret << std::endl;
        return ret;
    }

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasSetStream failed. ERROR: " << ret << std::endl;
        return ret;
    }

    float *aDevice = nullptr;
    float *xDevice = nullptr;
    float *alphaDevice = nullptr;
    size_t aByteSize = N * lda * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc(reinterpret_cast<void **>(&aDevice), aByteSize,
                                  ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc aDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMalloc(reinterpret_cast<void **>(&xDevice), xByteSize,
                         ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc xDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMalloc(reinterpret_cast<void **>(&alphaDevice), sizeof(float),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc alphaDevice failed." << std::endl; return aclRet; }

    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy aDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy xDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(alphaDevice, sizeof(float), &alphaVal, sizeof(float),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy alphaDevice failed." << std::endl; return aclRet; }

    ret = aclblasSsyr(handle, ACLBLAS_UPPER, N, alphaDevice, xDevice, incx, aDevice, lda);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasSsyr failed. ERROR: " << ret << std::endl;
        return ret;
    }

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtSynchronizeStream failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(A.data(), aByteSize, aDevice, aByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy A failed." << std::endl; return aclRet; }

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(alphaDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return static_cast<int>(VerifyResult(A, AGolden));
}

static int TestSsyrLower()
{
    std::cout << "Testing syr A2 (uplo=L):" << std::endl;

    int32_t deviceId = 0;
    constexpr int N = 128;
    constexpr int lda = N;
    constexpr int incx = 1;
    constexpr float alphaVal = 2.0f;

    std::vector<float> A(N * lda, 1.0f);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[i * lda + j] = 0.0f;
        }
        A[i * lda + i] = 3.0f;
    }
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i + 1);
    }
    std::vector<float> AGolden = A;

    ComputeGoldenSsyrLower(AGolden.data(), x.data(), alphaVal, N, lda);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasCreate failed. ERROR: " << ret << std::endl;
        return ret;
    }

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasSetStream failed. ERROR: " << ret << std::endl;
        return ret;
    }

    float *aDevice = nullptr;
    float *xDevice = nullptr;
    float *alphaDevice = nullptr;
    size_t aByteSize = N * lda * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc(reinterpret_cast<void **>(&aDevice), aByteSize,
                                  ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc aDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMalloc(reinterpret_cast<void **>(&xDevice), xByteSize,
                         ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc xDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMalloc(reinterpret_cast<void **>(&alphaDevice), sizeof(float),
                         ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMalloc alphaDevice failed." << std::endl; return aclRet; }

    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy aDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy xDevice failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(alphaDevice, sizeof(float), &alphaVal, sizeof(float),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy alphaDevice failed." << std::endl; return aclRet; }

    ret = aclblasSsyr(handle, ACLBLAS_LOWER, N, alphaDevice, xDevice, incx, aDevice, lda);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasSsyr failed. ERROR: " << ret << std::endl;
        return ret;
    }

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtSynchronizeStream failed." << std::endl; return aclRet; }
    aclRet = aclrtMemcpy(A.data(), aByteSize, aDevice, aByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) { std::cout << "aclrtMemcpy A failed." << std::endl; return aclRet; }

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(alphaDevice);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return static_cast<int>(VerifyResult(A, AGolden));
}

int32_t main(int32_t argc, char *argv[])
{
    int ret1 = TestSsyrUpper();
    int ret2 = TestSsyrLower();

    if (ret1 == 0 && ret2 == 0) {
        std::cout << "[PASS] syr_a2_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] syr_a2_test" << std::endl;
        return 1;
    }
}
