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
* \file sscal_test.cpp
* \brief Test for aclblasSscal and aclblasCsscal interfaces
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

uint32_t VerifySscalResult(std::vector<float> &output, std::vector<float> &golden)
{
    auto printTensor = [](std::vector<float> &tensor, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");
    constexpr float EPSILON = 1e-5f;
    for (size_t i = 0; i < output.size(); i++) {
        if (std::abs(output[i] - golden[i]) > EPSILON) {
            std::cout << "[Failed] Sscal accuracy verification failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Sscal accuracy verification passed." << std::endl;
    return 0;
}

uint32_t VerifyCsscalResult(std::vector<std::complex<float>> &output, std::vector<std::complex<float>> &golden)
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
            std::cout << "[Failed] Csscal Index " << i << ": output=(" << output[i].real() << "," << output[i].imag() 
                      << ") golden=(" << golden[i].real() << "," << golden[i].imag() << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Csscal accuracy verification passed." << std::endl;
    return 0;
}

int32_t TestSscal(aclblasHandle handle)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    constexpr float alpha = 2.5f;
    std::vector<float> x(totalLength, valueX);
    int64_t incx = 1;

    std::cout << "========== Testing aclblasSscal ==========" << std::endl;
    auto ret = aclblasSscal(handle, x.data(), alpha, totalLength, incx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSscal failed. ERROR: %d\n", ret); return ret);

    std::vector<float> golden(totalLength, valueX * alpha);
    return VerifySscalResult(x, golden);
}

int32_t TestCsscal(aclblasHandle handle)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr std::complex<float> valueX(1.2f, 0.5f);
    constexpr float alpha = 2.5f;
    std::vector<std::complex<float>> x(totalLength, valueX);
    int64_t incx = 1;

    std::cout << "========== Testing aclblasCsscal ==========" << std::endl;
    auto ret = aclblasCsscal(handle, x.data(), alpha, totalLength, incx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCsscal failed. ERROR: %d\n", ret); return ret);

    std::vector<std::complex<float>> golden(totalLength, valueX * alpha);
    return VerifyCsscalResult(x, golden);
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
    
    result = TestSscal(handle);
    if (result != 0) {
        std::cout << "[FAIL] Sscal test failed" << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return result;
    }
    std::cout << "[PASS] Sscal test passed" << std::endl;

    result = TestCsscal(handle);
    if (result != 0) {
        std::cout << "[FAIL] Csscal test failed" << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return result;
    }
    std::cout << "[PASS] Csscal test passed" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: 2 - Sscal Csscal" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}