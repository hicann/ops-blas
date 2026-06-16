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
 * \file aclblas_handle_internal.h
 * \brief Internal ops-blas handle structure (not exposed to users).
 */

#pragma once

#include <acl/acl.h>
#include <cstddef>

#include "cann_ops_blas_common.h"

/** @brief Default initial workspace size: 4 MiB. */
constexpr size_t ACLBLAS_DEFAULT_WORKSPACE_SIZE = 4U * 1024U * 1024U;

/**
 * @brief Internal ops-blas handle structure.
 *
 * Maintains default workspace (allocated/freed by the library) and user workspace
 * (allocated/freed by the user). The use_user_workspace flag selects which one is active.
 *
 * This structure is visible only inside implementation files. Public APIs use void* / void**
 * as the handle type.
 */
struct _aclblas_handle {
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;

    /* ========== Default workspace (library managed) ========== */
    void* default_workspace = nullptr;
    size_t default_workspace_size = 0;

    /* ========== User workspace (user managed) ========== */
    void* user_workspace = nullptr;
    size_t user_workspace_size = 0;

    bool use_user_workspace = false;
};

/**
 * @brief Returns the pointer of the currently active workspace.
 */
inline void* aclblasGetEffectiveWorkspace(const _aclblas_handle* h)
{
    if (h == nullptr) {
        return nullptr;
    }
    return h->use_user_workspace ? h->user_workspace : h->default_workspace;
}

/**
 * @brief Returns the size in bytes of the currently active workspace.
 */
inline size_t aclblasGetEffectiveWorkspaceSize(const _aclblas_handle* h)
{
    if (h == nullptr) {
        return 0;
    }
    return h->use_user_workspace ? h->user_workspace_size : h->default_workspace_size;
}

/**
 * @brief Switches back to the default workspace without clearing cached user workspace fields.
 */
inline void aclblasResetToDefaultWorkspace(_aclblas_handle* h)
{
    if (h != nullptr) {
        h->use_user_workspace = false;
    }
}
