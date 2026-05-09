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
* \file colwise_mul_test.cpp
* \brief Test for colwise_mul operator
*/

#include <cstdint>
#include <iostream>
#include <vector>
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

uint32_t VerifyResult(const float* output, const float* golden, size_t size, const char* test_name)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    
    const float epsilon = 1e-4;
    uint32_t errors = 0;
    
    for (size_t i = 0; i < size; i++) {
        if (std::abs(output[i] - golden[i]) > epsilon) {
            if (errors < 5) {
                std::cout << "Mismatch at index " << i << ": output=" << output[i] 
                          << ", golden=" << golden[i] << std::endl;
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] " << test_name << " verification failed with " << errors << " errors!" << std::endl;
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

    // Test: 2x3 complex matrix multiplied by 2-element complex vector
    // Matrix (complex):
    // Row 0: (1+2i, 3+4i, 5+6i)
    // Row 1: (7+8i, 9+10i, 11+12i)
    // Vector (complex):
    // vec[0] = 2+3i
    // vec[1] = 4+5i
    // Result:
    // Row 0: (2+3i) * (1+2i, 3+4i, 5+6i) = (-4+7i, -6+17i, -8+27i)
    // Row 1: (4+5i) * (7+8i, 9+10i, 11+12i) = (-12+67i, -14+85i, -16+103i)

    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    
    std::vector<float> mat = {
        1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,
        7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f
    };
    
    std::vector<float> vec = {
        2.0f, 3.0f,
        4.0f, 5.0f
    };
    
    std::vector<float> result(m * n * 2);
    
    std::vector<float> golden = {
        -4.0f, 7.0f,   -6.0f, 17.0f,  -8.0f, 27.0f,
        -12.0f, 67.0f, -14.0f, 85.0f, -16.0f, 103.0f
    };

    uint8_t *matDevice = nullptr;
    uint8_t *vecDevice = nullptr;
    uint8_t *resultDevice = nullptr;
    size_t matByteSize = mat.size() * sizeof(float);
    size_t vecByteSize = vec.size() * sizeof(float);
    size_t resultByteSize = result.size() * sizeof(float);

    aclError aclRet = aclrtMalloc((void **)&matDevice, matByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc matDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&vecDevice, vecByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc vecDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&resultDevice, resultByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(matDevice, matByteSize, mat.data(), matByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy matDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(vecDevice, vecByteSize, vec.data(), vecByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy vecDevice failed. ERROR: %d\n", aclRet); return aclRet);

    aclblasStatus_t statusRet = aclblasColwiseMul(handle, m, n, matDevice, vecDevice, resultDevice);
    CHECK_RET(statusRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasColwiseMul failed. ERROR: %d\n", statusRet); return statusRet);
    
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(result.data(), resultByteSize, resultDevice, resultByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    int32_t testRet = VerifyResult(result.data(), golden.data(), m * n * 2, "ColwiseMul Complex Test");

    aclrtFree(matDevice);
    aclrtFree(vecDevice);
    aclrtFree(resultDevice);

    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (testRet == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return testRet;
}