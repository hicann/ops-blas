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
#include "error_check.h"

uint32_t VerifyResult(float output, float golden)
{
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

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

int32_t test_scnrm2()
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 8 * 1024;
    constexpr float realVal = 1.5f;
    constexpr float imagVal = 2.0f;

    std::vector<std::complex<float>> x(n, std::complex<float>(realVal, imagVal));
    float result = 0.0f;
    int64_t incx = 1;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    uint8_t* xDevice = nullptr;
    uint8_t* resultDevice = nullptr;
    size_t inputByteSize = n * sizeof(std::complex<float>);
    size_t outputByteSize = sizeof(float);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasScnrm2(handle, n, xDevice, incx, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasScnrm2 failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    float golden = std::sqrt(n * (realVal * realVal + imagVal * imagVal));

    return VerifyResult(result, golden);
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    std::vector<float> x(totalLength, valueX);
    float result = 0.0f;
    int64_t incx = 1;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* xDevice = nullptr;
    float* resultDevice = nullptr;
    size_t inputByteSize = totalLength * sizeof(float);
    size_t outputByteSize = sizeof(float);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasSnrm2(handle, totalLength, xDevice, incx, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSnrm2 failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    float golden = std::abs(valueX) * std::sqrt(static_cast<float>(totalLength));

    int snrm2_result = VerifyResult(result, golden);
    if (snrm2_result != 0) {
        return snrm2_result;
    }

    std::cout << "\n========== Testing scnrm2 ==========\n" << std::endl;
    return test_scnrm2();
}