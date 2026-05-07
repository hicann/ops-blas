/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCBLAS_KERNEL_UTILS_H
#define ASCBLAS_KERNEL_UTILS_H

#include "ascblas_type.h"

static constexpr int64_t BIT_4 = 4;
static constexpr int64_t BIT_8 = 8;

__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    return 1 | (mode << BIT_4) | (flagId << BIT_8);
}

#include "ascblas_fp32_utils.h"
#endif