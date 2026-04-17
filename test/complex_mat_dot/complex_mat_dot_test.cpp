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
* \file complex_mat_dot_test.cpp
* \brief Test for complex matrix dot product
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
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

uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
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

    // Use relative error for floating point comparison
    constexpr float epsilon = 1e-5f;
    size_t errorCount = 0;
    for (size_t i = 0; i < output.size(); i++) {
        float relError = std::abs(output[i] - golden[i]) / (std::abs(golden[i]) + 1e-10f);
        if (relError > epsilon) {
            errorCount++;
        }
    }

    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed! Error count: " << errorCount << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    // Test with 4x4 complex matrix
    constexpr uint32_t m = 4;
    constexpr uint32_t n = 4;
    constexpr uint32_t complexSize = m * n * 2;  // 2 floats per complex number

    // Initialize input matrices with simple values
    // matx: all elements are (1.0 + 2.0i)
    // maty: all elements are (3.0 + 4.0i)
    std::vector<float> matx(complexSize);
    std::vector<float> maty(complexSize);
    std::vector<float> result(complexSize, 0.0f);

    for (uint32_t i = 0; i < m * n; i++) {
        matx[i * 2] = 1.0f;      // real part
        matx[i * 2 + 1] = 2.0f;  // imaginary part
        maty[i * 2] = 3.0f;      // real part
        maty[i * 2 + 1] = 4.0f;  // imaginary part
    }

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasComplexMatDot(matx.data(), maty.data(), result.data(), m, n, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasComplexMatDot failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // Calculate golden result: (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    std::vector<float> golden(complexSize);
    for (uint32_t i = 0; i < m * n; i++) {
        golden[i * 2] = matx[i * 2] * maty[i * 2] - matx[i * 2 + 1] * maty[i * 2 + 1];      // real part: 1*3 - 2*4 = -5
        golden[i * 2 + 1] = matx[i * 2] * maty[i * 2 + 1] + matx[i * 2 + 1] * maty[i * 2];  // imaginary part: 1*4 + 2*3 = 10
    }

    return VerifyResult(result, golden);
}
