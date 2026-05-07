/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in the compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
* \file sdot_test.cpp
* \brief Test for real vector dot product
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

uint32_t VerifyResult(float output, float golden)
{
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    float diff = std::abs(output - golden);
    float maxVal = std::max(std::abs(output), std::abs(golden));
    if (maxVal > 0 && diff / maxVal > 1e-5) {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 8 * 1024;

    std::vector<float> x(n);
    std::vector<float> y(n);
    float result = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float golden = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        golden += x[i] * y[i];
    }

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    int64_t incx = 1;
    int64_t incy = 1;
    auto ret = aclblasSdot(stream, x.data(), y.data(), &result, n, incx, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSdot failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "Testing sdot:" << std::endl;
    return VerifyResult(result, golden);
}