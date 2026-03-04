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
* \file scopy_test.cpp
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
    if (std::equal(output.begin(), output.end(), golden.begin())) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
    return 0;
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    constexpr float valueY = 2.3f;
    std::vector<float> x(totalLength, valueX);
    std::vector<float> y(totalLength, valueY);
    int64_t incx = 1;
    int64_t incy = 1;

    size_t totalByteSize = totalLength * sizeof(float);

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasScopy(x.data(), y.data(), totalLength, incx, incy, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasScopy failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::vector<float> golden(totalLength, valueX);
    return VerifyResult(y, golden);
}