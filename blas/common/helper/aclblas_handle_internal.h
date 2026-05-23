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
 * \brief ops-blas handle 内部结构与版本宏定义（不对外暴露）。
 */

#pragma once

#include <acl/acl.h>
#include <cstddef>

#include "cann_ops_blas_common.h"

/* ========== 版本信息（仅内部使用） ========== */
#define ACLBLAS_VERSION_MAJOR 1
#define ACLBLAS_VERSION_MINOR 0
#define ACLBLAS_VERSION_PATCH 0
#define ACLBLAS_VERSION_STRING "1.0.0"

/**
 * @brief 将版本号 (major, minor, patch) 编码为整型。
 *
 * 编码规则：MAJOR * 10000 + MINOR * 100 + PATCH，
 * 例如 1.0.0 -> 10000、1.2.3 -> 10203。
 */
#define ACLBLAS_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define ACLBLAS_VERSION \
    ACLBLAS_MAKE_VERSION(ACLBLAS_VERSION_MAJOR, ACLBLAS_VERSION_MINOR, ACLBLAS_VERSION_PATCH)

/**
 * @brief ops-blas handle 内部结构体
 *
 * 仅包含 kernel 执行所需的最小化字段：
 *   - stream：当前 stream，用于 kernel 异步执行
 *   - workspace / workspaceSize：内部 workspace，供算子临时存储
 *
 * 该结构体定义仅在实现文件中可见，对外部用户完全不可见。
 * 对外接口使用 void* / void** 作为 handle 类型。
 */
struct _aclblas_handle {
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;     ///< 当前 stream，用于 kernel 执行

    /* ========== 工作内存 ========== */
    void *workspace = nullptr;        ///< 内部 workspace（device 内存，由用户负责分配/释放）
    size_t workspaceSize = 0;         ///< workspace 大小（字节）
};
