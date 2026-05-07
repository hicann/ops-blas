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
* \file cswap_test.cpp
* \brief
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

uint32_t VerifyResult(float *outputX, float *outputY,
                      float *goldenX, float *goldenY, size_t totalFloats)
{
    auto printTensor = [](float *tensor, size_t size, const char *name) {
        constexpr size_t maxPrintSize = 10;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(size, maxPrintSize); i++) {
            std::cout << tensor[i] << " ";
        }
        if (size > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(outputX, totalFloats, "OutputX");
    printTensor(goldenX, totalFloats, "GoldenX");
    printTensor(outputY, totalFloats, "OutputY");
    printTensor(goldenY, totalFloats, "GoldenY");
    constexpr float EPSILON = 1e-3f;
    for (size_t i = 0; i < totalFloats; i++) {
        float diff = std::abs(outputX[i] - goldenX[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] X Index " << i << ": output=" << outputX[i] << " golden=" << goldenX[i] << " diff=" << diff << std::endl;
            return 1;
        }
    }
    for (size_t i = 0; i < totalFloats; i++) {
        float diff = std::abs(outputY[i] - goldenY[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Y Index " << i << ": output=" << outputY[i] << " golden=" << goldenY[i] << " diff=" << diff << std::endl;
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
    constexpr std::complex<float> valueX(1.5f, 0.5f);
    constexpr std::complex<float> valueY(2.5f, 1.5f);
    std::vector<std::complex<float>> x(totalLength, valueX);
    std::vector<std::complex<float>> y(totalLength, valueY);
    int64_t incx = 1;
    int64_t incy = 1;

    aclrtStream stream = nullptr;
    aclblasHandle handle;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    handle = stream;

    auto ret = aclblasCswap(handle, reinterpret_cast<float*>(x.data()), reinterpret_cast<float*>(y.data()), totalLength, incx, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCswap failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::vector<std::complex<float>> goldenX(totalLength, valueY);
    std::vector<std::complex<float>> goldenY(totalLength, valueX);
    return VerifyResult(reinterpret_cast<float*>(x.data()), reinterpret_cast<float*>(y.data()),
                        reinterpret_cast<float*>(goldenX.data()), reinterpret_cast<float*>(goldenY.data()),
                        totalLength * 2);
}