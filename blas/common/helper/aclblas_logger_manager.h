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
 * \file aclblas_logger_manager.h
 * \brief ops-blas log配置管理模块（不对外暴露）。
 */

#pragma once

#include <acl/acl.h>
#include <cstddef>

#include "cann_ops_blas.h"
#include "log/log.h"

#ifndef OP_MODULE_ID
#define OP_MODULE_ID 63
#endif

namespace AclBlas {

struct _aclblas_logger_configure {
    int logIsOn = 0;
    int logToStdOut = 0;
    int logToStdErr = 0;
    const char* logFile = nullptr;
    aclblasLogCallback userCallback = nullptr;
};

aclblasStatus_t aclblasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFile);
aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback);
aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback);

}

