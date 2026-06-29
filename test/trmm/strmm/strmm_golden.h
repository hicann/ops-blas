/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_GOLDEN_H
#define STRMM_GOLDEN_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline CBLAS_SIDE ToCblasSide(aclblasSideMode_t side)
{
    return (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
}

inline aclblasStatus_t aclblasStrmm_cpu(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t transA, aclblasDiagType_t diag,
    int64_t m, int64_t n, const float* alpha, const float* A,
    int64_t lda, float* B, int64_t ldb)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || A == nullptr || B == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    cblas_strmm(CblasRowMajor, ToCblasSide(side), ToCblasUplo(uplo),
        ToCblasOp(transA), ToCblasDiag(diag),
        static_cast<int>(m), static_cast<int>(n),
        *alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb));
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // STRMM_GOLDEN_H
