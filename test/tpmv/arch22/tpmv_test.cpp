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
* \file tpmv_test.cpp
* \brief
*/

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
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

static size_t PackedLowerIndex(uint32_t row, uint32_t col)
{
    return static_cast<size_t>(col + (row * (row + 1U)) / 2U);
}

static std::vector<float> BuildGolden(const std::vector<float> &aPacked, const std::vector<float> &x,
    uint32_t n, int64_t incx)
{
    std::vector<float> golden(n, 0.0f);
    for (uint32_t row = 0; row < n; ++row) {
        float acc = 0.0f;
        for (uint32_t col = 0; col <= row; ++col) {
            acc += aPacked[PackedLowerIndex(row, col)] * x[col * incx];
        }
        golden[row] = acc;
    }

    return golden;
}

static uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    std::cout << std::fixed << std::setprecision(6);

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

    constexpr float absTol = 1e-5f;
    constexpr float relTol = 1e-5f;
    auto closeEnough = [&](float a, float b) {
        float diff = std::abs(a - b);
        float scale = std::max(std::abs(a), std::abs(b));
        return diff <= absTol || diff <= relTol * scale;
    };

    for (size_t i = 0; i < output.size(); ++i) {
        if (!closeEnough(output[i], golden[i])) {
            std::cout << "[Failed] Case accuracy is verification failed at index " << i << " (" << output[i]
                      << " vs " << golden[i] << ")" << std::endl;
            return 1;
        }
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static int RunCase(uint32_t n)
{
    int32_t deviceId = 0;
    aclError ret = ACL_SUCCESS;
    std::mt19937 rng(20260325U + n);
    std::uniform_real_distribution<float> dist(0.0f, 0.5f);
    const int64_t incx = 1;

    std::vector<float> x(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }

    const size_t packedSize = static_cast<size_t>(n) * (static_cast<size_t>(n) + 1U) / 2U;
    std::vector<float> aPacked(packedSize, 0.0f);
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col <= row; ++col) {
            aPacked[PackedLowerIndex(row, col)] = dist(rng);
        }
    }

    std::vector<float> golden = BuildGolden(aPacked, x, n, incx);
    std::vector<float> y(n, 0.0f);

    aclrtStream stream = nullptr;

    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    ret = aclblasTpmv(aPacked.data(), x.data(), y.data(), n, incx, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasTpmv failed. ERROR: %d\n", ret); return ret);

    int status = VerifyResult(y, golden);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return status;
}

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;
    int ret = RunCase(4096);
    if (ret != 0) {
        return ret;
    }
    return RunCase(5000);
}