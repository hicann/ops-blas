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

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden)
{
    auto printTensor = [](std::vector<float>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(
            tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
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

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;

    constexpr uint32_t m = 4;
    constexpr uint32_t n = 4;
    constexpr uint32_t complexSize = m * n * 2;

    std::vector<float> matx(complexSize);
    std::vector<float> maty(complexSize);
    std::vector<float> result(complexSize, 0.0f);

    for (uint32_t i = 0; i < m * n; i++) {
        matx[i * 2] = 1.0f;
        matx[i * 2 + 1] = 2.0f;
        maty[i * 2] = 3.0f;
        maty[i * 2 + 1] = 4.0f;
    }

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    size_t dataSize = complexSize * sizeof(float);

    uint8_t* matxDevice = nullptr;
    uint8_t* matyDevice = nullptr;
    uint8_t* resultDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&matxDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc matxDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&matyDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc matyDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(matxDevice, dataSize, matx.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy matxDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(matyDevice, dataSize, maty.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy matyDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(resultDevice, dataSize, result.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy resultDevice failed. ERROR: %d\n", aclRet); return aclRet);

    auto ret = aclblasComplexMatDot(handle, m, n, matxDevice, matyDevice, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasComplexMatDot failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(result.data(), dataSize, resultDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(matxDevice);
    aclrtFree(matyDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::vector<float> golden(complexSize);
    for (uint32_t i = 0; i < m * n; i++) {
        golden[i * 2] = matx[i * 2] * maty[i * 2] - matx[i * 2 + 1] * maty[i * 2 + 1];
        golden[i * 2 + 1] = matx[i * 2] * maty[i * 2 + 1] + matx[i * 2 + 1] * maty[i * 2];
    }

    return VerifyResult(result, golden);
}
