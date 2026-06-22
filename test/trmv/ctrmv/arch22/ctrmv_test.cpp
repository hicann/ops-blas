/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
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

uint32_t VerifyResult(
    const float* output, const float* golden, size_t size, const char* test_name, float epsilon = 1e-3f)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    size_t printCnt = size < 8 ? size : 8;
    std::cout << "output: ";
    for (size_t i = 0; i < printCnt; ++i)
        std::cout << output[i] << " ";
    std::cout << std::endl;
    std::cout << "golden: ";
    for (size_t i = 0; i < printCnt; ++i)
        std::cout << golden[i] << " ";
    std::cout << std::endl;
    uint32_t errors = 0;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(output[i] - golden[i]) > epsilon) {
            if (errors < 5) {
                std::cout << "Mismatch at index " << i << ": output=" << output[i] << ", golden=" << golden[i]
                          << std::endl;
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

void ComputeCtrmvGolden(
    const float* A, const float* x, float* golden, int64_t n, int64_t lda, int64_t incx, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag)
{
    for (int64_t i = 0; i < n; i++) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int64_t j = 0; j < n; j++) {
            std::complex<float> a_op(0.0f, 0.0f);
            if (trans == ACLBLAS_OP_N) {
                bool isValid = (uplo == ACLBLAS_LOWER) ? (j <= i) : (j >= i);
                if (isValid) {
                    int64_t aIdx = j * lda + i;
                    a_op = std::complex<float>(A[aIdx * 2], A[aIdx * 2 + 1]);
                    if (diag == ACLBLAS_UNIT && i == j)
                        a_op = std::complex<float>(1.0f, 0.0f);
                }
            } else if (trans == ACLBLAS_OP_T) {
                bool isValidT = (uplo == ACLBLAS_LOWER) ? (i <= j) : (i >= j);
                if (isValidT) {
                    int64_t aIdxT = i * lda + j;
                    a_op = std::complex<float>(A[aIdxT * 2], A[aIdxT * 2 + 1]);
                    if (diag == ACLBLAS_UNIT && i == j)
                        a_op = std::complex<float>(1.0f, 0.0f);
                }
            } else {
                bool isValidC = (uplo == ACLBLAS_LOWER) ? (i <= j) : (i >= j);
                if (isValidC) {
                    int64_t aIdxC = i * lda + j;
                    a_op = std::conj(std::complex<float>(A[aIdxC * 2], A[aIdxC * 2 + 1]));
                    if (diag == ACLBLAS_UNIT && i == j)
                        a_op = std::complex<float>(1.0f, 0.0f);
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

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto blasRet = aclblasCreate(&handle);
    CHECK_RET(
        blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    blasRet = aclblasSetStream(handle, stream);
    CHECK_RET(
        blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

    int ret = 0;

    constexpr int64_t n = 4;
    constexpr int64_t lda = 4;
    constexpr int64_t incx = 1;

    std::vector<float> A(n * lda * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            int64_t aIdx = j * lda + i;
            A[aIdx * 2] = 1.0f;
            A[aIdx * 2 + 1] = 2.0f;
        }
    }

    std::vector<float> x(n * incx * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        x[i * incx * 2] = static_cast<float>(i + 0.1);
        x[i * incx * 2 + 1] = static_cast<float>(i + 1.1);
    }

    std::vector<float> golden(n * incx * 2, 0.0f);
    ComputeCtrmvGolden(A.data(), x.data(), golden.data(), n, lda, incx, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT);

    uint8_t* aDevice = nullptr;
    uint8_t* xDevice = nullptr;
    size_t aByteSize = n * lda * 2 * sizeof(float);
    size_t xByteSize = n * incx * 2 * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCtrmv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, aDevice, lda, xDevice, incx);
    if (blasRet == ACLBLAS_STATUS_SUCCESS) {
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); ret = 1);
        aclRet = aclrtMemcpy(x.data(), xByteSize, xDevice, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); ret = 1);
        ret |= VerifyResult(x.data(), golden.data(), n * incx * 2, "Ctrmv Lower N NonUnit (n=4)");
    } else {
        LOG_PRINT("aclblasCtrmv failed. ERROR: %d\n", blasRet);
        ret = 1;
    }

    aclrtFree(aDevice);
    aclrtFree(xDevice);

    std::vector<float> A2(n * lda * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            int64_t aIdx = j * lda + i;
            A2[aIdx * 2] = 1.0f;
            A2[aIdx * 2 + 1] = 2.0f;
        }
    }

    std::vector<float> x2(n * incx * 2, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        x2[i * incx * 2] = static_cast<float>(i + 0.1);
        x2[i * incx * 2 + 1] = static_cast<float>(i + 1.1);
    }

    std::vector<float> golden2(n * incx * 2, 0.0f);
    ComputeCtrmvGolden(A2.data(), x2.data(), golden2.data(), n, lda, incx, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT);

    uint8_t* a2Device = nullptr;
    uint8_t* x2Device = nullptr;
    aclRet = aclrtMalloc((void**)&a2Device, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc a2Device failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&x2Device, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x2Device failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(a2Device, aByteSize, A2.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy a2Device failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x2Device, xByteSize, x2.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x2Device failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCtrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, n, a2Device, lda, x2Device, incx);
    if (blasRet == ACLBLAS_STATUS_SUCCESS) {
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); ret = 1);
        aclRet = aclrtMemcpy(x2.data(), xByteSize, x2Device, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x2 failed. ERROR: %d\n", aclRet); ret = 1);
        ret |= VerifyResult(x2.data(), golden2.data(), n * incx * 2, "Ctrmv Upper N Unit (n=4)");
    } else {
        LOG_PRINT("aclblasCtrmv failed. ERROR: %d\n", blasRet);
        ret = 1;
    }

    aclrtFree(a2Device);
    aclrtFree(x2Device);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

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