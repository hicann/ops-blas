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
 * \file host_utils.h
 * \brief Shared host-side utility functions for blasLt.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <securec.h>
#include <type_traits>

inline aclblasStatus_t CheckedMemcpyS(void* dest, size_t destMax, const void* src, size_t count)
{
    if (memcpy_s(dest, destMax, src, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CheckedMemsetS(void* dest, size_t destMax, size_t count)
{
    if (memset_s(dest, destMax, 0, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool MemcpySSucceeds(void* dest, size_t destMax, const void* src, size_t count)
{
    return memcpy_s(dest, destMax, src, count) == EOK;
}

template <typename T>
inline aclblasStatus_t AllocHandle(T** out)
{
    if (out == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *out = nullptr;
    T* p = new (std::nothrow) T();
    if (p == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    *out = p;
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline aclblasStatus_t FreeHandle(T* p)
{
    if (p == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    delete p;
    return ACLBLAS_STATUS_SUCCESS;
}

inline size_t GetTypeSize(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_INT8:
            return 1;
        default:
            return 0;
    }
}

inline bool IsDataTypeSupported(aclDataType dt)
{
    return GetTypeSize(dt) != 0;
}

inline bool IsMxfp8Type(aclDataType dt)
{
    return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2;
}

inline bool IsMxfp4Type(aclDataType dt)
{
    return dt == ACL_FLOAT4_E2M1;
}

inline bool CheckComputeTypeCompatibility(aclblasComputeType_t ct, aclDataType typeA, aclDataType typeB)
{
    if (typeA == ACL_FLOAT && typeB == ACL_FLOAT) {
        return ct == ACLBLAS_COMPUTE_32F || ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_TF32;
    }
    if (IsMxfp8Type(typeA) && IsMxfp8Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    if (IsMxfp4Type(typeA) && IsMxfp4Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    if (typeA == ACL_FLOAT16 && typeB == ACL_FLOAT16) {
        return ct == ACLBLAS_COMPUTE_16F || ct == ACLBLAS_COMPUTE_16F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F ||
               ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_16F;
    }
    if (typeA == ACL_INT8 && typeB == ACL_INT8) {
        return ct == ACLBLAS_COMPUTE_32I || ct == ACLBLAS_COMPUTE_32I_PEDANTIC;
    }
    return false;
}

template <typename T1, typename T2>
inline uint64_t CeilDivU64(T1 a, T2 b)
{
    const uint64_t ua = static_cast<uint64_t>(a);
    const uint64_t ub = static_cast<uint64_t>(b);
    if (ua == 0U || ub == 0U) {
        return 0U;
    }
    return (ua - 1U) / ub + 1U;
}

template <typename R = uint32_t, typename T1, typename T2>
inline R CeilDiv(T1 a, T2 b)
{
    static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
                  "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_arithmetic<R>::value, "CeilDiv return type must be arithmetic");
    return static_cast<R>(CeilDivU64(a, b));
}

template <typename R = uint32_t, typename T1, typename T2>
inline R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}
