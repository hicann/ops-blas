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
* \file sswap_test.cpp
* \brief
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
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
    printTensor(outputX, "OutputX");
    printTensor(goldenX, "GoldenX");
    printTensor(outputY, "OutputY");
    printTensor(goldenY, "GoldenY");
    constexpr float EPSILON = 1e-5f;
    for (size_t i = 0; i < outputX.size(); i++) {
        if (std::abs(outputX[i] - goldenX[i]) > EPSILON) {
            std::cout << "[Failed] X accuracy is verification failed!" << std::endl;
            return 1;
        }
    }
    for (size_t i = 0; i < outputY.size(); i++) {
        if (std::abs(outputY[i] - goldenY[i]) > EPSILON) {
            std::cout << "[Failed] Y accuracy is verification failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.5f;
    constexpr float valueY = 2.5f;
    std::vector<float> x(totalLength, valueX);
    std::vector<float> y(totalLength, valueY);
    int64_t incx = 1;
    int64_t incy = 1;

    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    handle = stream;

    auto ret = aclblasSswap(handle, x.data(), y.data(), totalLength, incx, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSswap failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::vector<float> goldenX(totalLength, valueY);
    std::vector<float> goldenY(totalLength, valueX);
    return VerifyResult(x, y, goldenX, goldenY);
}