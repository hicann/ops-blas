/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLAS_TEST_H
#define BLAS_TEST_H

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

namespace blas_test_detail {

struct AclGuard {
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    ~AclGuard()
    {
        try {
            if (stream != nullptr) {
                aclrtDestroyStream(stream);
                stream = nullptr;
            }
            if (handle != nullptr) {
                aclblasDestroy(handle);
                handle = nullptr;
            }
            aclrtResetDevice(TEST_DEVICE_ID);
            aclFinalize();
        } catch (...) {
        }
    }
};

inline AclGuard& globalAcl()
{
    static AclGuard guard;
    return guard;
}

} // namespace blas_test_detail

template <typename ParamType>
class BlasTest : public ::testing::TestWithParam<ParamType> {
protected:
    static void SetUpTestSuite()
    {
        auto& guard = blas_test_detail::globalAcl();
        if (guard.handle == nullptr) {
            aclError initRet = aclInit(nullptr);
            ASSERT_TRUE(initRet == ACL_SUCCESS || initRet == ACL_ERROR_REPEAT_INITIALIZE)
                << "aclInit failed with error: " << initRet;
            ASSERT_EQ(aclrtSetDevice(TEST_DEVICE_ID), ACL_SUCCESS);
            ASSERT_EQ(aclblasCreate(&guard.handle), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtCreateStream(&guard.stream), ACL_SUCCESS);
            ASSERT_EQ(aclblasSetStream(guard.handle, guard.stream), ACLBLAS_STATUS_SUCCESS);
        }
        handle_ = guard.handle;
        stream_ = guard.stream;
    }
    static void TearDownTestSuite() {}

    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

template <typename ParamType>
aclblasHandle_t BlasTest<ParamType>::handle_ = nullptr;

template <typename ParamType>
aclrtStream BlasTest<ParamType>::stream_ = nullptr;

#endif // BLAS_TEST_H
