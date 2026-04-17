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
* \file csrot_test.cpp
* \brief Test for complex vector rotation
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

uint32_t VerifyResult(std::vector<float> &outputX, std::vector<float> &outputY,
                      std::vector<float> &goldenX, std::vector<float> &goldenY)
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
    printTensor(outputX, "Output X");
    printTensor(goldenX, "Golden X");
    printTensor(outputY, "Output Y");
    printTensor(goldenY, "Golden Y");

    // Use relative error for floating point comparison
    constexpr float epsilon = 1e-5f;
    size_t errorCount = 0;
    
    for (size_t i = 0; i < outputX.size(); i++) {
        float relError = std::abs(outputX[i] - goldenX[i]) / (std::abs(goldenX[i]) + 1e-10f);
        if (relError > epsilon) {
            errorCount++;
        }
    }
    
    for (size_t i = 0; i < outputY.size(); i++) {
        float relError = std::abs(outputY[i] - goldenY[i]) / (std::abs(goldenY[i]) + 1e-10f);
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

    // Test with 1024 elements
    constexpr uint32_t n = 1024;
    
    // Rotation parameters: c = cos(θ), s = sin(θ) where θ = 45°
    constexpr float theta = M_PI / 4;  // 45 degrees
    constexpr float c = 0.7071067811865476;  // cos(45°) ≈ 0.707
    constexpr float s = 0.7071067811865476;  // sin(45°) ≈ 0.707

    // Initialize input vectors
    std::vector<float> x(n, 1.0f);  // x = [1, 1, 1, ...]
    std::vector<float> y(n, 2.0f);  // y = [2, 2, 2, ...]

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasCsrot(x.data(), y.data(), n, c, s, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCsrot failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // Calculate golden result
    // x[i] = c*x[i] + s*y[i] = 0.707*1 + 0.707*2 = 2.121
    // y[i] = c*y[i] - s*x[i] = 0.707*2 - 0.707*1 = 0.707
    std::vector<float> goldenX(n, c * 1.0f + s * 2.0f);
    std::vector<float> goldenY(n, c * 2.0f - s * 1.0f);

    return VerifyResult(x, y, goldenX, goldenY);
}
