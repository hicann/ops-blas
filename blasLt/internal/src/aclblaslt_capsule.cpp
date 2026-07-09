/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclblaslt_capsule.cpp
 * \brief Impl-into-capsule packing helper implementation.
 */

#include "aclblaslt_capsule.h"

#include "host_utils.h"

#include <cstddef>

aclblasStatus_t MatPackTransformImpl(void* capsule, size_t capsuleBytes, const void* impl, size_t implBytes)
{
    aclblasStatus_t copyStatus = CheckedMemcpyS(capsule, capsuleBytes, impl, implBytes);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (capsuleBytes > implBytes) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(capsule) + implBytes, capsuleBytes - implBytes, capsuleBytes - implBytes);
    }
    return copyStatus;
}
