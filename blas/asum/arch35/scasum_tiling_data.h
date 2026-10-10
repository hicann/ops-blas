/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "common/helper/kernel_constant.h"

// Maximum AIV cores available on a single ascend950 (arch35) chip.
// Used as an upper bound to cap tiling data arrays.
constexpr uint32_t SCASUM_MAX_CORE_NUM = 64;

struct ScasumTilingData {
    int64_t n;                                 // number of complex elements
    int64_t incx;                              // complex stride
    uint32_t useCoreNum;
    uint32_t startOffset[SCASUM_MAX_CORE_NUM]; // complex-element start offset per block
    uint32_t calNum[SCASUM_MAX_CORE_NUM];      // complex-element count per block
    uint32_t nthreads;
};
