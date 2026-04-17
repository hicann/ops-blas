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

    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    // Test parameters
    constexpr int64_t m = 4;  // rows
    constexpr int64_t n = 4;  // columns (complex elements)
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;
    constexpr int64_t lda = m;  // leading dimension
    
    // Alpha: complex scalar
    std::complex<float> alpha(1.0f, 0.0f);  // alpha = 1.0 + 0.0i
    
    // Initialize x vector (m complex elements)
    std::vector<std::complex<float>> xHost(m);
    for (int64_t i = 0; i < m; i++) {
        xHost[i] = std::complex<float>(i + 1.0f, i + 0.5f);
    }
    
    // Initialize y vector (n complex elements)
    std::vector<std::complex<float>> yHost(n);
    for (int64_t i = 0; i < n; i++) {
        yHost[i] = std::complex<float>(i + 2.0f, i + 1.5f);
    }
    
    // Initialize A matrix (m x n complex elements)
    std::vector<std::complex<float>> AHost(m * n);
    for (int64_t i = 0; i < m * n; i++) {
        AHost[i] = std::complex<float>(i * 0.1f, i * 0.2f);
    }
    
    // Compute golden result
    std::vector<std::complex<float>> goldenHost = AHost;
    computeGolden(goldenHost, xHost, yHost, alpha, m, n);
    
    // Call cgerc
    ret = aclblasCgerc(m, n, alpha, xHost.data(), incx, yHost.data(), incy, AHost.data(), lda, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgerc failed. ERROR: %d\n", ret); 
              aclrtDestroyStream(stream); aclFinalize(); return ret);
    
    aclrtDestroyStream(stream);
    aclFinalize();
    
    return VerifyResult(AHost, goldenHost, "cgerc_test");
}
