/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMV_BATCHED_TILING_DATA_H
#define GEMV_BATCHED_TILING_DATA_H

#include <cstdint>

constexpr uint8_t GEMV_BATCHED_BUFFER_NUM = 2;
constexpr uint32_t GEMV_BATCHED_TMPBUF_NUM = 2;
constexpr uint32_t GEMV_BATCHED_ELENUM_LINE_ALIGNED = 32;
constexpr uint32_t GEMV_BATCHED_UBUF_SIZE = 190 * 1024;
constexpr uint32_t GEMV_BATCHED_BYTENUM_BLOCK = 32;
constexpr uint32_t GEMV_BATCHED_BYTENUM_REPEAT = 256;

constexpr uint32_t GEMV_BATCHED_ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t GEMV_BATCHED_ELENUM_BLOCK_FP16 = 16;

// Non-DB mode buffer counts
constexpr uint32_t GEMV_BATCHED_BN_NORMAL = 1;
constexpr uint32_t GEMV_BATCHED_TN_NORMAL = 1;

// 分核逻辑: 除最后一个 core 外，每个 core 处理 batchPerCore 个 batch；
//          最后一个 core 处理 batchTail 个（余数）。
// startMatId = coreIdx * batchPerCore
// calMatNum  = (coreIdx == usedCoreNum - 1) ? batchTail : batchPerCore
#pragma pack(push, 4)
struct GemvBatchedTilingData {
    uint32_t dtype;
    uint32_t trans;
    float    alpha;
    float    beta;
    uint32_t m;
    uint32_t n;
    uint32_t batchGroupSize;
    uint32_t coreNum;
    uint32_t usedCoreNum;
    uint32_t batchCount;
    uint32_t batchPerCore;
    uint32_t batchTail;
    uint32_t nTile;
    uint32_t mTile;
    uint32_t bufInA;
    uint32_t bufInx;
    uint32_t bufInY;
    uint32_t bufOut;
    uint32_t bufMatTmp;
    uint32_t bufVecTmp;
    int32_t  lda;
    int32_t  incx;
    int32_t  incy;
};
#pragma pack(pop)

#endif  // GEMV_BATCHED_TILING_DATA_H
