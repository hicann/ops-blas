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
* \file cdot_test.cpp
* \brief Test for complex dot product (cdotu and cdotc)
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

uint32_t VerifyResult(float *output, float *golden, uint32_t len)
{
    auto printTensor = [](float *tensor, uint32_t size, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        for (uint32_t i = 0; i < std::min(size, (uint32_t)maxPrintSize); i++) {
            std::cout << tensor[i] << " ";
        }
        if (size > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, len, "Output");
    printTensor(golden, len, "Golden");

    bool pass = true;
    for (uint32_t i = 0; i < len; i++) {
        float diff = std::abs(output[i] - golden[i]);
        float maxVal = std::max(std::abs(output[i]), std::abs(golden[i]));
        if (maxVal > 0 && diff / maxVal > 1e-5) {
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
}

void ComputeCdotuGolden(const std::vector<float>& x, const std::vector<float>& y, 
                        uint32_t complexNum, float* golden)
{
    float goldenReal = 0.0f;
    float goldenImag = 0.0f;
    
    for (uint32_t i = 0; i < complexNum; i++) {
        float xReal = x[i * 2];
        float xImag = x[i * 2 + 1];
        float yReal = y[i * 2];
        float yImag = y[i * 2 + 1];
        
        goldenReal += xReal * yReal - xImag * yImag;
        goldenImag += xReal * yImag + xImag * yReal;
    }
    
    golden[0] = goldenReal;
    golden[1] = goldenImag;
}

void ComputeCdotcGolden(const std::vector<float>& x, const std::vector<float>& y, 
                        uint32_t complexNum, float* golden)
{
    float goldenReal = 0.0f;
    float goldenImag = 0.0f;
    
    for (uint32_t i = 0; i < complexNum; i++) {
        float xReal = x[i * 2];
        float xImag = x[i * 2 + 1];
        float yReal = y[i * 2];
        float yImag = y[i * 2 + 1];
        
        goldenReal += xReal * yReal + xImag * yImag;
        goldenImag += xReal * yImag - xImag * yReal;
    }
    
    golden[0] = goldenReal;
    golden[1] = goldenImag;
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 256;
    constexpr uint32_t complexNum = n / 2;
    int64_t incx = 1;
    int64_t incy = 1;

    std::vector<float> x(n);
    std::vector<float> y(n);
    float result[2] = {0.0f, 0.0f};
    float golden[2] = {0.0f, 0.0f};

    for (uint32_t i = 0; i < complexNum; i++) {
        x[i * 2] = (float)(1);
        x[i * 2 + 1] = (float)(0.5);
        y[i * 2] = (float)(3);
        y[i * 2 + 1] = (float)(2);
    }

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclError aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", aclRet); aclblasDestroy(handle); return aclRet);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *resultDevice = nullptr;
    size_t inputByteSize = n * sizeof(float);
    size_t outputByteSize = 2 * sizeof(float);

    aclRet = aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMalloc((void **)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(yDevice, inputByteSize, y.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);

    std::cout << "=== Testing aclblasCdotu ===" << std::endl;
    ComputeCdotuGolden(x, y, complexNum, golden);
    
    ret = aclblasCdotu(handle, complexNum, xDevice, incx, yDevice, incy, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCdotu failed. ERROR: %d\n", ret); return ret);
    
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    
    uint32_t cdotuResult = VerifyResult(result, golden, 2);

    std::cout << "\n=== Testing aclblasCdotc ===" << std::endl;
    ComputeCdotcGolden(x, y, complexNum, golden);
    
    ret = aclblasCdotc(handle, complexNum, xDevice, incx, yDevice, incy, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCdotc failed. ERROR: %d\n", ret); return ret);
    
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    
    uint32_t cdotcResult = VerifyResult(result, golden, 2);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(resultDevice);

    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return cdotuResult + cdotcResult;
}