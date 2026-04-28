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
 * \file aclblas_logger.cpp
 * \brief ops-blas log配置管理模块
 */

#include "aclblas_logger_manager.h"

namespace {
static _aclblas_logger_configure configure;
}

#ifdef __cplusplus
extern "C" {
#endif

aclblasStatus_t aclblasLoggerConfigure(const char* logFile, bool logToStdOut, bool logToKdlls, aclblasLogLevel_t logLevel)
{
    configure.logFile = logFile;
    configure.logToStdOut = logToStdOut;
    configure.logToKdlls = logToKdlls;
    configure.logLevel = logLevel;
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback)
{
    if (handle == nullptr) {
        return aclblasStatus_t::ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    configure.userCallback = userCallback;
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback)
{
    if (handle == nullptr) {
        return aclblasStatus_t::ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    userCallback = configure.userCallback;
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif