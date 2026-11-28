/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "aclblaslt_matmul_problem.h"
#include "matmul_regular_kernel.h"

inline LtRegularMatmulType GetRegularMatmulType(aclDataType a, aclDataType b, aclDataType c, aclDataType d)
{
    if (c != d) {
        return LtRegularMatmulType::NONE;
    }
    if (a == ACL_FLOAT16 && b == ACL_FLOAT16 && d == ACL_FLOAT16) {
        return LtRegularMatmulType::FP16;
    }
    if (a == ACL_BF16 && b == ACL_BF16 && d == ACL_BF16) {
        return LtRegularMatmulType::BF16;
    }
    if (a == ACL_INT8 && b == ACL_INT8 && d == ACL_INT32) {
        return LtRegularMatmulType::INT8;
    }
    if (d == ACL_FLOAT16) {
        if (a == ACL_FLOAT8_E4M3FN && b == ACL_FLOAT8_E4M3FN) {
            return LtRegularMatmulType::FP8_E4M3;
        }
        if (a == ACL_FLOAT8_E5M2 && b == ACL_FLOAT8_E5M2) {
            return LtRegularMatmulType::FP8_E5M2;
        }
        if (a == ACL_FLOAT8_E4M3FN && b == ACL_FLOAT8_E5M2) {
            return LtRegularMatmulType::FP8_E4M3_E5M2;
        }
        if (a == ACL_FLOAT8_E5M2 && b == ACL_FLOAT8_E4M3FN) {
            return LtRegularMatmulType::FP8_E5M2_E4M3;
        }
    }
    return LtRegularMatmulType::NONE;
}

inline bool HasRegularMatmulPrecision(LtRegularMatmulType type, aclblasComputeType_t compute, aclDataType scale)
{
    return type == LtRegularMatmulType::INT8 ? compute == ACLBLAS_COMPUTE_32I && scale == ACL_INT32 :
        compute == ACLBLAS_COMPUTE_32F && scale == ACL_FLOAT;
}

inline bool HasRegularMatmulScalars(const aclblasLtMatmulDescImpl* desc, LtRegularMatmulType type)
{
    return HasRegularMatmulPrecision(type, desc->computeType, desc->scaleType) &&
        desc->pointerMode == ACLBLASLT_POINTER_MODE_HOST && desc->scaleA == nullptr && desc->scaleB == nullptr;
}

bool CanTileRegularMatmul(
    LtRegularMatmulType type, uint64_t m, uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
    uint64_t ldc, uint64_t ldd);
size_t RegularMatmulWorkspaceSize(uint64_t m, uint64_t n);
bool CanTileRegularMatmulLayouts(
    LtRegularMatmulType type, const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a,
    const aclblasLtMatrixLayoutImpl* b, const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d);
aclblasStatus_t MatmulRegularLaunch(const MatmulProblem& problem, LtRegularMatmulType type);
