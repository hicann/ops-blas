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
* \file ctrmv_test.cpp
* \brief Test for ctrmv operator
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

uint32_t VerifyResult(const float* output, const float* golden, size_t size, const char* test_name, float epsilon = 1e-3f)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    // Print partial output and golden values for inspection
    size_t printCnt = size < 8 ? size : 8;
    std::cout << "output: ";
    for (size_t i = 0; i < printCnt; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
    std::cout << "golden: ";
    for (size_t i = 0; i < printCnt; ++i) std::cout << golden[i] << " ";
    std::cout << std::endl;
    uint32_t errors = 0;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(output[i] - golden[i]) > epsilon) {
            if (errors < 5) {
                std::cout << "Mismatch at index " << i << ": output=" << output[i] 
                          << ", golden=" << golden[i] << std::endl;
            }
            errors++;
        }
    }
    if (errors == 0) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] " << test_name << " verification failed with " << errors << " errors!" << std::endl;
        return 1;
    }
}

// Golden computation for ctrmv: x_out = op(A) * x
// A is stored in column-major (Fortran) order as complex64 (pairs of float)
// In column-major: A[i][j] is at index j * lda + i
void ComputeCtrmvGolden(const float *A, const float *x, float *golden,
                         int64_t n, int64_t lda, int64_t incx,
                         aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    for (int64_t i = 0; i < n; i++) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int64_t j = 0; j < n; j++) {
            std::complex<float> a_op(0.0f, 0.0f);
            if (trans == ACLBLAS_OP_N) {
                bool isValid = (uplo == ACLBLAS_LOWER) ? (j <= i) : (j >= i);
                if (isValid) {
                    int64_t aIdx = j * lda + i;  // column-major: A[i][j] at j*lda+i
                    a_op = std::complex<float>(A[aIdx * 2], A[aIdx * 2 + 1]);
                    if (diag == ACLBLAS_UNIT && i == j) a_op = std::complex<float>(1.0f, 0.0f);
                }
            } else if (trans == ACLBLAS_OP_T) {
                bool isValidT = (uplo == ACLBLAS_LOWER) ? (i <= j) : (i >= j);
                if (isValidT) {
                    int64_t aIdxT = i * lda + j;  // column-major, transposed: A[j][i] at i*lda+j
                    a_op = std::complex<float>(A[aIdxT * 2], A[aIdxT * 2 + 1]);
                    if (diag == ACLBLAS_UNIT && i == j) a_op = std::complex<float>(1.0f, 0.0f);
                }
            } else {  // ACLBLAS_OP_C
                bool isValidC = (uplo == ACLBLAS_LOWER) ? (i <= j) : (i >= j);
                if (isValidC) {
                    int64_t aIdxC = i * lda + j;  // column-major, conjugate transposed: A[j][i] at i*lda+j
                    a_op = std::conj(std::complex<float>(A[aIdxC * 2], A[aIdxC * 2 + 1]));
                    if (diag == ACLBLAS_UNIT && i == j) a_op = std::complex<float>(1.0f, 0.0f);
                }
            }
            int64_t xIdx = j * incx;
            std::complex<float> x_val(x[xIdx * 2], x[xIdx * 2 + 1]);
            sum += a_op * x_val;
        }
        int64_t outIdx = i * incx;
        golden[outIdx * 2] = sum.real();
        golden[outIdx * 2 + 1] = sum.imag();
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 1;
    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    aclblasHandle handle;

    handle = stream;

    int ret = 0;

    // Test case: Lower triangular, N (non-transpose), Non-Unit diagonal
    // Following BLAS convention: A is stored in column-major (Fortran) order
    // In column-major: A[i][j] is at index j * lda + i
    constexpr int64_t n = 4;
    constexpr int64_t lda = 4;
    constexpr int64_t incx = 1;

    // Create lower triangular matrix A in column-major (Fortran) order
    // A(i,j) = (1+2i) for j <= i, 0 otherwise
    std::vector<float> A(n * lda * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            int64_t aIdx = j * lda + i;  // column-major: A[i][j] at j*lda+i
            A[aIdx * 2] = 1.0f;          // real = 1
            A[aIdx * 2 + 1] = 2.0f;      // imag = 2
        }
    }

    // Create x vector: x[i] = (i, i) (matching sip example)
    std::vector<float> x(n * incx * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        x[i * incx * 2] = static_cast<float>(i + 0.1);      // real
        x[i * incx * 2 + 1] = static_cast<float>(i + 1.1);  // imag
    }

    // Compute golden
    std::vector<float> golden(n * incx * 2, 0.0f);
    ComputeCtrmvGolden(A.data(), x.data(), golden.data(), n, lda, incx,
                       ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT);

    // Run NPU computation
    std::vector<float> x_npu = x;
    std::vector<float> A_npu = A;
    auto aclRet = aclblasCtrmv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                                n, A_npu.data(), lda, x_npu.data(), incx);
    if (aclRet == ACL_SUCCESS) {
        ret |= VerifyResult(x_npu.data(), golden.data(), n * incx * 2, "Ctrmv Lower N NonUnit (n=4)");
    } else {
        LOG_PRINT("aclblasCtrmv failed. ERROR: %d\n", aclRet);
        ret = 1;
    }

    // Test case 2: Upper triangular, N, Unit diagonal
    // A is stored in column-major (Fortran) order
    std::vector<float> A2(n * lda * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            int64_t aIdx = j * lda + i;  // column-major: A[i][j] at j*lda+i
            A2[aIdx * 2] = 1.0f;         // real = 1
            A2[aIdx * 2 + 1] = 2.0f;     // imag = 2
        }
    }

    std::vector<float> x2(n * incx * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        x2[i * incx * 2] = static_cast<float>(i + 0.1);      // real
        x2[i * incx * 2 + 1] = static_cast<float>(i + 1.1);  // imag
    }

    std::vector<float> golden2(n * incx * 2, 0.0f);
    ComputeCtrmvGolden(A2.data(), x2.data(), golden2.data(), n, lda, incx,
                       ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT);

    std::vector<float> x2_npu = x2;
    std::vector<float> A2_npu = A2;
    aclRet = aclblasCtrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT,
                           n, A2_npu.data(), lda, x2_npu.data(), incx);
    if (aclRet == ACL_SUCCESS) {
        ret |= VerifyResult(x2_npu.data(), golden2.data(), n * incx * 2, "Ctrmv Upper N Unit (n=4)");
    } else {
        LOG_PRINT("aclblasCtrmv failed. ERROR: %d\n", aclRet);
        ret = 1;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    handle = nullptr;
    
    if (ret == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    return ret;
}
