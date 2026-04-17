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
* \file sasum_test.cpp
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

    const float epsilon = 1e-4;
    if (std::abs(output - golden) < epsilon) {
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

    constexpr uint32_t totalLength = 128;
    constexpr float valueX = 1.2f;
    std::vector<float> x(totalLength, valueX);
    int64_t incx = 1;
    float result = 0.0f;

    size_t totalByteSize = totalLength * sizeof(float);

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasSasum(x.data(), &result, totalLength, incx, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSasum failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    float golden = totalLength * std::abs(valueX);
    return VerifyResult(result, golden);
}
