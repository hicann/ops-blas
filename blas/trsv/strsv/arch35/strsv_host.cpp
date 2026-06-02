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
 * \file strsv_host.cpp
 * \brief Single-precision trsv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "strsv_tiling_data.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static constexpr uint32_t SIMT_OPTIMAL_THREADS = 128;

static aclblasStatus_t ValidateTrsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx, int lda, const float* A,
    const float* x)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max<int>(1, n), return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CHECK_RET(h != nullptr, return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateTrsvParams(uplo, trans, diag, n, incx, lda, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t numThreads = SIMT_OPTIMAL_THREADS;

    StrsvTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int32_t>(incx);
    tiling.lda = static_cast<int32_t>(lda);
    tiling.numThreads = numThreads;

    strsv_kernel_do(reinterpret_cast<GM_ADDR>(const_cast<float*>(A)), reinterpret_cast<GM_ADDR>(x), tiling, h->stream);

    aclError aclRet = aclrtSynchronizeStream(h->stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return ACLBLAS_STATUS_INTERNAL_ERROR);

    return ACLBLAS_STATUS_SUCCESS;
}
