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
 * \file aclblas_auxiliary.cpp
 * \brief ops-blas handle 生命周期、stream、workspace、版本信息等辅助函数实现。
 */

#include "cann_ops_blas.h"
#include "aclblas_handle_internal.h"

#include <acl/acl.h>
#include <new>

namespace {

/**
 * @brief 将对外 void* 句柄安全转换为内部 _aclblas_handle*。
 */
inline _aclblas_handle *ToInternal(aclblasHandle_t handle)
{
    return reinterpret_cast<_aclblas_handle *>(handle);
}

} // namespace

extern "C" {

aclblasStatus_t aclblasCreate(aclblasHandle_t *handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // 关键：防止重复创建覆盖已有指针导致的内存泄漏。
    if (*handle != nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto *h = new (std::nothrow) _aclblas_handle();
    if (h == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    *handle = reinterpret_cast<void *>(h);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasDestroy(aclblasHandle_t handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);

    // stream 和 workspace 由用户托管，此处仅清理句柄自身字段，不释放 device 内存。
    h->stream = nullptr;
    h->workspace = nullptr;
    h->workspaceSize = 0;

    delete h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);
    h->stream = stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream *stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (stream == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    *stream = h->stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void *workspace, size_t workspaceSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // 只允许两种合法组合：
    //   (workspace != nullptr, workspaceSize > 0) —— 用户提供 workspace
    //   (workspace == nullptr, workspaceSize == 0) —— 清空 workspace 设置
    const bool hasBuffer = (workspace != nullptr);
    const bool hasSize = (workspaceSize != 0);
    if (hasBuffer != hasSize) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    h->workspace = workspace;
    h->workspaceSize = workspaceSize;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int *version)
{
    if (version == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *version = ACLBLAS_VERSION;
    return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"
