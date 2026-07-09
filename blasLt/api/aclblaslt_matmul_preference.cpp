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
 * \file aclblaslt_matmul_preference.cpp
 * \brief Public C API: matmul heuristic-search preference CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <algorithm>
#include <cstdint>
#include <new>

extern "C" {

aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *pref = nullptr;
    auto* capsule = new (std::nothrow) aclblasLtMatmulPreferenceOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t copyStatus = CheckedMemsetS(capsule, sizeof(*capsule), sizeof(*capsule));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    aclblasLtMatmulPreferenceImpl impl;
    copyStatus = PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    *pref = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(pref);
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (pref == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE: {
            if (sizeInBytes != sizeof(uint32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            uint32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(v));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            // 0=heuristic, 1=exhaustive, 2=fast
            if (v > 2) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.searchMode = v;
            break;
        }

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: {
            if (sizeInBytes != sizeof(size_t) && sizeInBytes != sizeof(uint64_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            size_t v = 0;
            const size_t copyBytes = std::min(sizeInBytes, sizeof(v));
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, copyBytes);
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            if (v > INT64_MAX) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.maxWorkspaceBytes = v;
            break;
        }

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    copyStatus = CheckedMemcpyS(pref, sizeof(*pref), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (pref == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t requiredSize = 0;
    const void* srcPtr = nullptr;

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE:
            requiredSize = sizeof(impl.searchMode);
            srcPtr = &impl.searchMode;
            break;

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            requiredSize = sizeof(impl.maxWorkspaceBytes);
            srcPtr = &impl.maxWorkspaceBytes;
            break;

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    return CopyAttributeOut(buf, sizeInBytes, srcPtr, requiredSize, sizeWritten);
}

} // extern "C"
