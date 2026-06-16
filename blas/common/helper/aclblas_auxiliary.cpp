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
 * \brief Auxiliary functions for ops-blas handle lifecycle, stream, workspace, and version info.
 */

#include "cann_ops_blas.h"
#include "aclblas_handle_internal.h"
#include "aclblas_version.h"

#include <acl/acl.h>
#include <algorithm>
#include <new>

namespace {

/**
 * @brief Safely casts a public void* handle to internal _aclblas_handle*.
 */
inline _aclblas_handle* toInternal(aclblasHandle_t handle) { return reinterpret_cast<_aclblas_handle*>(handle); }

/**
 * @brief Frees the default workspace device memory held by the handle.
 */
void freeDefaultWorkspace(_aclblas_handle* h)
{
    if (h == nullptr || h->default_workspace == nullptr) {
        return;
    }
    aclrtFree(h->default_workspace);
    h->default_workspace = nullptr;
    h->default_workspace_size = 0;
}

/**
 * @brief Allocates a default workspace of the given size for the handle.
 */
aclblasStatus_t allocateDefaultWorkspace(_aclblas_handle* h, size_t size)
{
    if (h == nullptr || size == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    void* ptr = nullptr;
    const aclError aclRet = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->default_workspace = ptr;
    h->default_workspace_size = size;
    return ACLBLAS_STATUS_SUCCESS;
}

/**
 * @brief Ensures the default workspace is at least @p requiredSize bytes.
 *
 * Synchronizes the handle stream before reallocating so in-flight kernels finish
 * using the old workspace. This sync occurs only on the rare expansion path.
 */
aclblasStatus_t ensureDefaultWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    if (h == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (h->use_user_workspace) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (requiredSize <= h->default_workspace_size) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (h->stream != nullptr) {
        const aclError aclRet = aclrtSynchronizeStream(h->stream);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    const size_t newSize = std::max(requiredSize, h->default_workspace_size * 2);
    freeDefaultWorkspace(h);
    return allocateDefaultWorkspace(h, newSize);
}

} // namespace

extern "C" {

aclblasStatus_t aclblasCreate(aclblasHandle_t* handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // Prevent overwriting an existing pointer to avoid memory leaks.
    if (*handle != nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = new (std::nothrow) _aclblas_handle();
    if (h == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t st = allocateDefaultWorkspace(h, ACLBLAS_DEFAULT_WORKSPACE_SIZE);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        delete h;
        return st;
    }

    *handle = reinterpret_cast<void*>(h);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasDestroy(aclblasHandle_t handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = toInternal(handle);

    // Step 1: synchronize the associated stream and wait for GPU work to finish.
    if (h->stream != nullptr) {
        const aclError aclRet = aclrtSynchronizeStream(h->stream);
        if (aclRet != ACL_SUCCESS) {
            freeDefaultWorkspace(h);
            h->user_workspace = nullptr;
            h->user_workspace_size = 0;
            h->use_user_workspace = false;
            h->stream = nullptr;
            delete h;
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    // Step 2: free the default workspace (library-owned resource).
    freeDefaultWorkspace(h);

    // Step 3: clear user workspace references without freeing user memory.
    h->user_workspace = nullptr;
    h->user_workspace_size = 0;
    h->use_user_workspace = false;
    h->stream = nullptr;

    delete h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = toInternal(handle);
    h->stream = stream;

    // Reset to the default workspace when switching streams (cuBLAS-compatible behavior).
    aclblasResetToDefaultWorkspace(h);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (stream == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = toInternal(handle);
    *stream = h->stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void* workspace, size_t workspaceSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = toInternal(handle);

    // workspace == nullptr: switch back to the default workspace.
    if (workspace == nullptr) {
        aclblasResetToDefaultWorkspace(h);
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (workspaceSize == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Grow-only policy: keep the current user workspace when the new size is not larger.
    if (h->use_user_workspace && workspaceSize <= h->user_workspace_size) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    h->user_workspace = workspace;
    h->user_workspace_size = workspaceSize;
    h->use_user_workspace = true;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetWorkspace(aclblasHandle_t handle, void** workspace, size_t* workspaceSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (workspace == nullptr || workspaceSize == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = toInternal(handle);
    *workspace = aclblasGetEffectiveWorkspace(h);
    *workspaceSize = aclblasGetEffectiveWorkspaceSize(h);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int* version)
{
    (void)handle;
    if (version == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *version = ACLBLAS_VERSION;
    return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"
