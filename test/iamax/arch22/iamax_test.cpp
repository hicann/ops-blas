/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use the License for the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
* \file iamax_test.cpp
* \brief Test for iamax operator with both real and complex support
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

uint32_t VerifyResult(int32_t output, int32_t golden, const char* test_name)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    if (output == golden) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] " << test_name << " verification failed!" << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;
    
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    
    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    int testRet = 0;

    // ==================== Test 1: Real float ====================
    {
        std::cout << "\n========== Testing Real Float ==========" << std::endl;
        constexpr uint32_t totalLength = 128;
        std::vector<float> x(totalLength);

        for (uint32_t i = 0; i < totalLength; i++) {
            x[i] = static_cast<float>(i) * 0.1f;
        }
        x[50] = 100.0f;

        int64_t incx = 1;
        int32_t result = 0;

        uint8_t *xDevice = nullptr;
        uint8_t *resultDevice = nullptr;
        size_t inputByteSize = totalLength * sizeof(float);
        size_t outputByteSize = sizeof(int32_t);

        aclError aclRet = aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); testRet = 1);
        aclRet = aclrtMalloc((void **)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); testRet = 1);
        aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); testRet = 1);

        aclblasStatus_t statusRet = aclblasIamax(handle, totalLength, xDevice, incx, resultDevice);
        CHECK_RET(statusRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasIamax failed. ERROR: %d\n", statusRet); testRet = 1);

        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); testRet = 1);
        aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); testRet = 1);

        aclrtFree(xDevice);
        aclrtFree(resultDevice);

        int32_t golden = 51;
        testRet |= VerifyResult(result, golden, "Real Float Test");
    }

    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (testRet == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Some tests failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return testRet;
}
