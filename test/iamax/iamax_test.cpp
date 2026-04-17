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
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    int ret = 0;

    // ==================== Test 1: Real float ====================
    {
        std::cout << "\n========== Testing Real Float ==========" << std::endl;
        constexpr uint32_t totalLength = 128;
        std::vector<float> x(totalLength);

        // Create test data: put the maximum absolute value at index 50
        for (uint32_t i = 0; i < totalLength; i++) {
            x[i] = static_cast<float>(i) * 0.1f;
        }
        x[50] = 100.0f;  // Maximum absolute value

        int64_t incx = 1;
        int32_t result = 0;
        uint32_t dtypeFlag = 0;  // 0 for real float

        auto aclRet = aclblasIamax(x.data(), &result, totalLength, incx, dtypeFlag, stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclblasIamax failed for real. ERROR: %d\n", aclRet); ret = 1);

        // Golden: index of maximum absolute value (1-based index as per BLAS convention)
        int32_t golden = 51;  // 50 + 1 (BLAS uses 1-based indexing)
        ret |= VerifyResult(result, golden, "Real Float Test");
    }

    // ==================== Test 2: Complex float ====================
    {
        std::cout << "\n========== Testing Complex Float ==========" << std::endl;
        constexpr uint32_t totalLength = 64;  // Number of complex elements
        // Complex numbers are stored as [real0, imag0, real1, imag1, ...]
        std::vector<float> x(totalLength * 2);  // 2 floats per complex number

        // Create test data: complex numbers with varying magnitudes
        for (uint32_t i = 0; i < totalLength; i++) {
            // Create complex number: (i*0.1, i*0.2)
            // Magnitude = sqrt((i*0.1)^2 + (i*0.2)^2) = i*sqrt(0.05)
            x[i * 2] = static_cast<float>(i) * 0.1f;      // Real part
            x[i * 2 + 1] = static_cast<float>(i) * 0.2f;  // Imaginary part
        }

        // Put maximum magnitude complex number at index 30
        // Complex number: (50, 60) with magnitude = sqrt(50^2 + 60^2) = 78.1
        x[30 * 2] = 50.0f;      // Real part
        x[30 * 2 + 1] = 60.0f;  // Imaginary part

        int64_t incx = 1;
        int32_t result = 0;
        uint32_t dtypeFlag = 1;  // 1 for complex float

        auto aclRet = aclblasIamax(x.data(), &result, totalLength, incx, dtypeFlag, stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclblasIamax failed for complex. ERROR: %d\n", aclRet); ret = 1);

        // Golden: index of maximum magnitude complex number (1-based index)
        int32_t golden = 31;  // 30 + 1 (BLAS uses 1-based indexing)
        ret |= VerifyResult(result, golden, "Complex Float Test");
    }

    // ==================== Test 3: Real float with negative values ====================
    {
        std::cout << "\n========== Testing Real Float with Negatives ==========" << std::endl;
        constexpr uint32_t totalLength = 100;
        std::vector<float> x(totalLength);

        // Create test data with negative values
        for (uint32_t i = 0; i < totalLength; i++) {
            x[i] = -static_cast<float>(i) * 0.5f;  // All negative
        }
        x[75] = -200.0f;  // Maximum absolute value (most negative)

        int64_t incx = 1;
        int32_t result = 0;
        uint32_t dtypeFlag = 0;  // Real float

        auto aclRet = aclblasIamax(x.data(), &result, totalLength, incx, dtypeFlag, stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclblasIamax failed for negative values. ERROR: %d\n", aclRet); ret = 1);

        int32_t golden = 76;  // 75 + 1
        ret |= VerifyResult(result, golden, "Real Float with Negatives Test");
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (ret == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Some tests failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return ret;
}
