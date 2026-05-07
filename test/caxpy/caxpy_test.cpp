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
* \file caxpy_test.cpp
* \brief Test for aclblasCaxpy interface
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
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

uint32_t VerifyCaxpyResult(std::vector<std::complex<float>> &output, std::vector<std::complex<float>> &golden)
{
    auto printTensor = [](std::vector<std::complex<float>> &tensor, const char *name) {
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
    constexpr float EPSILON = 1e-3f;
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output[i] - golden[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Caxpy Index " << i << ": output=(" << output[i].real() << "," << output[i].imag() 
                      << ") golden=(" << golden[i].real() << "," << golden[i].imag() << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Caxpy accuracy verification passed." << std::endl;
    return 0;
}

int32_t TestCaxpy(aclblasHandle handle)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr std::complex<float> valueX(1.0f, 0.5f);
    constexpr std::complex<float> valueY(2.0f, 1.0f);
    constexpr std::complex<float> alpha(2.0f, 1.0f);
    std::vector<std::complex<float>> x(totalLength, valueX);
    std::vector<std::complex<float>> y(totalLength, valueY);
    int64_t incx = 1;
    int64_t incy = 1;

    std::cout << "========== Testing aclblasCaxpy ==========" << std::endl;
    std::cout << "Formula: y = alpha * x + y" << std::endl;
    std::cout << "alpha = (" << alpha.real() << ", " << alpha.imag() << ")" << std::endl;
    std::cout << "x = (" << valueX.real() << ", " << valueX.imag() << ") * " << totalLength << std::endl;
    std::cout << "y = (" << valueY.real() << ", " << valueY.imag() << ") * " << totalLength << std::endl;

    auto ret = aclblasCaxpy(handle, x.data(), y.data(), alpha, totalLength, incx, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCaxpy failed. ERROR: %d\n", ret); return ret);

    std::vector<std::complex<float>> golden(totalLength);
    for (size_t i = 0; i < totalLength; i++) {
        golden[i] = alpha * x[i] + valueY;
    }

    return VerifyCaxpyResult(y, golden);
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    handle = stream;

    int32_t result = 0;
    
    result = TestCaxpy(handle);
    if (result != 0) {
        std::cout << "[FAIL] Caxpy test failed" << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return result;
    }
    std::cout << "[PASS] Caxpy test passed" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: 1 - Caxpy" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}