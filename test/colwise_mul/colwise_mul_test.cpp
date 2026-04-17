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
            if (errors < 5) {  // Only print first 5 errors
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
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    int ret = 0;

    // Test: 2x3 complex matrix multiplied by 2-element complex vector
    // Matrix (complex, stored as [real0, imag0, real1, imag1, ...]):
    // Row 0: (1+2i, 3+4i, 5+6i)
    // Row 1: (7+8i, 9+10i, 11+12i)
    // Vector (complex):
    // vec[0] = 2+3i
    // vec[1] = 4+5i
    // Result:
    // Row 0: (2+3i) * (1+2i, 3+4i, 5+6i) = (-4+7i, -6+17i, -8+27i)
    // Row 1: (4+5i) * (7+8i, 9+10i, 11+12i) = (-12+67i, -14+85i, -16+103i)

    constexpr int64_t m = 2;  // rows
    constexpr int64_t n = 3;  // columns (complex elements)
    
    // Matrix: 2 rows, 3 complex columns
    // Stored as: [r0_real0, r0_imag0, r0_real1, r0_imag1, r0_real2, r0_imag2, r1_real0, ...]
    std::vector<float> mat = {
        1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,   // Row 0
        7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f  // Row 1
    };
    
    // Vector: 2 complex elements
    std::vector<float> vec = {
        2.0f, 3.0f,  // vec[0] = 2+3i
        4.0f, 5.0f   // vec[1] = 4+5i
    };
    
    std::vector<float> result(m * n * 2);  // Result matrix
    
    auto aclRet = aclblasColwiseMul(mat.data(), vec.data(), result.data(), m, n, stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclblasColwiseMul failed. ERROR: %d\n", aclRet); ret = 1);
    
    // Expected result
    std::vector<float> golden = {
        -4.0f, 7.0f,   -6.0f, 17.0f,  -8.0f, 27.0f,   // Row 0
        -12.0f, 67.0f, -14.0f, 85.0f, -16.0f, 103.0f  // Row 1
    };
    
    ret |= VerifyResult(result.data(), golden.data(), m * n * 2, "ColwiseMul Complex Test");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (ret == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return ret;
}
