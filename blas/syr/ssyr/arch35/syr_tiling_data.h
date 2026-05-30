/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SYR_TILING_DATA_H
#define SYR_TILING_DATA_H

#include <cstdint>

constexpr uint32_t SYR_MAX_CORE_NUM = 50;

struct SyrTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    uint32_t useCoreNum;
    uint32_t rowStride;
    uint32_t startRow[SYR_MAX_CORE_NUM];
    uint32_t rowCount[SYR_MAX_CORE_NUM];
};

#endif  // SYR_TILING_DATA_H
