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
* \file cgerc_test.cpp
* \brief Test for cgerc operator
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
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

uint32_t VerifyResult(const std::vector<std::complex<float>>& output, 
                      const std::vector<std::complex<float>>& golden, 
                      const char* test_name)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    
    // Print first few results for comparison (horizontal format)
    constexpr size_t maxPrintCount = 5;
    std::cout << "\n--- First " << maxPrintCount << " results ---" << std::endl;
    std::cout << "Index  | Output (real, imag)       | Golden (real, imag)        | Diff" << std::endl;
    std::cout << "-------|---------------------------|----------------------------|--------" << std::endl;
    for (size_t i = 0; i < std::min(output.size(), maxPrintCount); i++) {
        printf("%-6zu | (%8.4f, %8.4f) | (%8.4f, %8.4f)  | %8.6f\n", 
               i, output[i].real(), output[i].imag(), 
               golden[i].real(), golden[i].imag(), 
               std::abs(output[i] - golden[i]));
    }
    
    const float epsilon = 1e-3;
    uint32_t errors = 0;
    float maxError = 0.0f;
    size_t maxErrorIndex = 0;
    
    for (size_t i = 0; i < output.size(); i++) {
        float error = std::abs(output[i] - golden[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIndex = i;
        }
        if (error > epsilon) {
            if (errors < 5) {  // Only print first 5 errors
                printf("Mismatch[%zu]: out=(%.4f,%.4f) gold=(%.4f,%.4f) diff=%.6f\n", 
                       i, output[i].real(), output[i].imag(), 
                       golden[i].real(), golden[i].imag(), error);
            }
            errors++;
        }
    }
    
    std::cout << "\n--- Statistics ---" << std::endl;
    printf("Total: %zu, MaxErr: %.6f @ idx %zu, Threshold: %.6f, Errors: %u\n", 
           output.size(), maxError, maxErrorIndex, epsilon, errors);
    
    if (errors == 0) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] " << test_name << " verification failed!" << std::endl;
        return 1;
    }
}

// Compute golden result for cgerc: A = alpha * x * conj(y^T) + A
void computeGolden(std::vector<std::complex<float>>& A,
                   const std::vector<std::complex<float>>& x,
                   const std::vector<std::complex<float>>& y,
                   const std::complex<float>& alpha,
                   int64_t m, int64_t n)
{
    // A = alpha * x * conj(y^T) + A
    // For each element A[i][j] = alpha * x[i] * conj(y[j]) + A[i][j]
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A[i * n + j] = alpha * x[i] * std::conj(y[j]) + A[i * n + j];
        }
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;

    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    constexpr int64_t m = 4;
    constexpr int64_t n = 4;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;
    constexpr int64_t lda = m;
    
    std::complex<float> alpha(1.0f, 0.0f);
    
    std::vector<std::complex<float>> xHost(m);
    for (int64_t i = 0; i < m; i++) {
        xHost[i] = std::complex<float>(i + 1.0f, i + 0.5f);
    }
    
    std::vector<std::complex<float>> yHost(n);
    for (int64_t i = 0; i < n; i++) {
        yHost[i] = std::complex<float>(i + 2.0f, i + 1.5f);
    }
    
    std::vector<std::complex<float>> AHost(m * n);
    for (int64_t i = 0; i < m * n; i++) {
        AHost[i] = std::complex<float>(i * 0.1f, i * 0.2f);
    }
    
    std::vector<std::complex<float>> goldenHost = AHost;
    computeGolden(goldenHost, xHost, yHost, alpha, m, n);
    
    size_t xSize = m * sizeof(std::complex<float>);
    size_t ySize = n * sizeof(std::complex<float>);
    size_t aSize = m * n * sizeof(std::complex<float>);
    
    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;
    uint8_t* ADevice = nullptr;
    
    aclError aclRet = aclrtMalloc((void**)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ADevice failed. ERROR: %d\n", aclRet); return aclRet);
    
    aclRet = aclrtMemcpy(xDevice, xSize, xHost.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, ySize, yHost.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(ADevice, aSize, AHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy ADevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgerc(handle, m, n, alpha, xDevice, incx, yDevice, incy, ADevice, lda);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgerc failed. ERROR: %d\n", ret); return ret);
    
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    
    aclRet = aclrtMemcpy(AHost.data(), aSize, ADevice, aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy AHost failed. ERROR: %d\n", aclRet); return aclRet);
    
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(ADevice);
    
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return VerifyResult(AHost, goldenHost, "cgerc_test");
}
