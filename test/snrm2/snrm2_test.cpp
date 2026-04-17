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
* \file snrm2_test.cpp
* \brief
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

    // Use relative error for floating point comparison
    float relError = std::abs(output - golden) / std::abs(golden);
    constexpr float epsilon = 1e-5f;

    if (relError < epsilon) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed! Relative error: " << relError << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    std::vector<float> x(totalLength, valueX);
    float result = 0.0f;
    int64_t incx = 1;

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasSnrm2(x.data(), &result, totalLength, incx, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSnrm2 failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // Calculate golden result: sqrt(n * x^2) = |x| * sqrt(n)
    float golden = std::abs(valueX) * std::sqrt(static_cast<float>(totalLength));

    return VerifyResult(result, golden);
}
