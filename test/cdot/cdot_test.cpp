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
* \file cdot_test.cpp
* \brief Test for complex dot product
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

uint32_t VerifyResult(float *output, float *golden, uint32_t len)
{
    auto printTensor = [](float *tensor, uint32_t size, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        for (uint32_t i = 0; i < std::min(size, (uint32_t)maxPrintSize); i++) {
            std::cout << tensor[i] << " ";
        }
        if (size > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, len, "Output");
    printTensor(golden, len, "Golden");

    bool pass = true;
    for (uint32_t i = 0; i < len; i++) {
        float diff = std::abs(output[i] - golden[i]);
        float maxVal = std::max(std::abs(output[i]), std::abs(golden[i]));
        if (maxVal > 0 && diff / maxVal > 1e-5) {
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 256;
    constexpr uint32_t complexNum = n / 2;

    std::vector<float> x(n);
    std::vector<float> y(n);
    float result[2] = {0.0f, 0.0f};

    for (uint32_t i = 0; i < complexNum; i++) {
        x[i * 2] = (float)(1);
        x[i * 2 + 1] = (float)(0.5);
        y[i * 2] = (float)(3);
        y[i * 2 + 1] = (float)(2);
    }

    float goldenReal = 0.0f;
    float goldenImag = 0.0f;
    int64_t isConj = 0;
    for (uint32_t i = 0; i < complexNum; i++) {
        float xReal = x[i * 2];
        float xImag = x[i * 2 + 1];
        float yReal = y[i * 2];
        float yImag = y[i * 2 + 1];
        
        goldenReal += xReal * yReal - xImag * yImag;
        goldenImag += xReal * yImag + xImag * yReal;
    }
    float golden[2] = {goldenReal, goldenImag};

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasCdot(x.data(), y.data(), result, complexNum, isConj, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCdot failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "Testing cdot (isConj=0):" << std::endl;
    return VerifyResult(result, golden, 2);
}
