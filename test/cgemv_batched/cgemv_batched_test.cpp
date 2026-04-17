/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"

using namespace std;

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

// Complex number multiplication
complex<float> complexMul(const complex<float>& a, const complex<float>& b) {
    return a * b;
}

// Compute golden result for cgemv: y = alpha * A * x + beta * y
void computeGolden(const vector<complex<float>>& A, const vector<complex<float>>& x,
                   vector<complex<float>>& y, int m, int n, int trans,
                   const complex<float>& alpha, const complex<float>& beta) {
    vector<complex<float>> yTemp(m);
    for (int i = 0; i < m; i++) {
        complex<float> sum(0.0f, 0.0f);
        for (int j = 0; j < n; j++) {
            if (trans == 0) {
                // Normal: y = A * x
                sum += A[i * n + j] * x[j];
            } else if (trans == 1) {
                // Transpose: y = A^T * x
                sum += A[j * m + i] * x[j];
            } else {
                // Conjugate transpose: y = A^H * x
                sum += conj(A[j * m + i]) * x[j];
            }
        }
        yTemp[i] = sum;
    }
    
    // Apply alpha and beta: y = alpha * yTemp + beta * y
    for (int i = 0; i < m; i++) {
        y[i] = alpha * yTemp[i] + beta * y[i];
    }
}

uint32_t VerifyResult(std::vector<complex<float>> &output, std::vector<complex<float>> &golden)
{
    auto printTensor = [](std::vector<complex<float>> &tensor, const char *name) {
        constexpr size_t maxPrintSize = 10;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(tensor.size(), maxPrintSize); i++) {
            std::cout << "(" << tensor[i].real() << "," << tensor[i].imag() << ") ";
        }
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    // Use relative error for floating point comparison
    constexpr float epsilon = 1e-3f;
    size_t errorCount = 0;
    float maxError = 0.0f;
    
    for (size_t i = 0; i < output.size(); i++) {
        float error = std::abs(output[i] - golden[i]);
        if (error > maxError) {
            maxError = error;
        }
        if (error > epsilon) {
            errorCount++;
        }
    }
    
    std::cout << "Max error: " << maxError << std::endl;
    
    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy verification failed." << std::endl;
        return 1;
    }
}

int main() {
    // Initialize ACL
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    ret = aclrtCreateContext(&context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtDestroyContext(context); aclFinalize(); return ret);

    // Test parameters
    const int64_t batchCount = 3;
    const int64_t m = 32;
    const int64_t n = 32;
    const int32_t trans = 0;  // Normal
    
    // Alpha and beta parameters
    complex<float> alpha(1.0f, 0.0f);  // alpha = 1.0 + 0.0i
    complex<float> beta(0.0f, 0.0f);   // beta = 0.0 + 0.0i
    
    // Leading dimension and increments
    const int64_t lda = m;  // For row-major storage
    const int64_t incx = 1;
    const int64_t incy = 1;

    // Allocate host memory using complex<float>
    vector<complex<float>> AHost(batchCount * m * n);
    vector<complex<float>> xHost(batchCount * n);
    vector<complex<float>> yHost(batchCount * m);
    vector<complex<float>> yGolden(batchCount * m);

    // Initialize test data
    for (int b = 0; b < batchCount; b++) {
        // Initialize matrix A (m x n complex)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = b * m * n + i * n + j;
                AHost[idx] = complex<float>((i + j + b) * 0.1f, (i - j + b) * 0.1f);
            }
        }

        // Initialize vector x (n complex)
        for (int i = 0; i < n; i++) {
            int idx = b * n + i;
            xHost[idx] = complex<float>((i + b) * 0.2f, (i - b) * 0.2f);
        }
    }

    // Compute golden result
    for (int b = 0; b < batchCount; b++) {
        vector<complex<float>> AComplex(m * n);
        vector<complex<float>> xComplex(n);
        vector<complex<float>> yComplex(m);

        // Copy data for this batch
        for (int i = 0; i < m * n; i++) {
            AComplex[i] = AHost[b * m * n + i];
        }
        for (int i = 0; i < n; i++) {
            xComplex[i] = xHost[b * n + i];
        }

        // Compute y = alpha * A * x + beta * y
        computeGolden(AComplex, xComplex, yComplex, m, n, trans, alpha, beta);

        // Copy result
        for (int i = 0; i < m; i++) {
            yGolden[b * m + i] = yComplex[i];
        }
    }

    // Call cgemv_batched
    ret = aclblasCgemvBatched(AHost.data(), xHost.data(), yHost.data(),
                              alpha, lda, beta, incx, incy,
                              batchCount, m, n, trans, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemvBatched failed. ERROR: %d\n", ret); 
              aclrtDestroyStream(stream); aclrtDestroyContext(context); aclFinalize(); return ret);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();

    return VerifyResult(yHost, yGolden);
}
