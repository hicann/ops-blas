/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSWAP_GOLDEN_H
#define SSWAP_GOLDEN_H

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Reference implementation — same signature as aclblasSswap.
// Parameter validation priority: handle → n<=0 → x/y null → inc==0 → inc<0
inline aclblasStatus_t aclblasSswap_cpu(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx < 0 || incy < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    for (int i = 0; i < n; i++) {
        std::swap(x[i * incx], y[i * incy]);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SSWAP_GOLDEN_H
