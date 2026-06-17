/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "sswap_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

class SswapAIV {
public:
    __aicore__ inline SswapAIV() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);

    TPipe* pipe_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueX_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueY_;
    SswapTilingData tiling_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void SswapAIV::Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingGm);

    // compute offset and count for this core (R4: no arrays, self-calculate)
    // perCoreN is ELEMENTS_PER_BLOCK-aligned, so offset is always aligned.
    // Last core absorbs the remainder (tail elements).
    myOffset_ = blockIdx_ * tiling_.perCoreN;
    myCount_ = tiling_.perCoreN;
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.remainder;
    }

    xGM_.SetGlobalBuffer((__gm__ float*)x, tiling_.totalN);
    yGM_.SetGlobalBuffer((__gm__ float*)y, tiling_.totalN);

    // Single Buffer: swap has no Vector stage, DB provides no benefit
    uint32_t bufSize = tiling_.tileSize * sizeof(float);
    pipe_->InitBuffer(inQueueX_, BUFFER_NUM, bufSize);
    pipe_->InitBuffer(inQueueY_, BUFFER_NUM, bufSize);
}

__aicore__ inline void SswapAIV::ParseTilingData(GM_ADDR tilingGm)
{
    __gm__ SswapTilingData* tilingPtr = reinterpret_cast<__gm__ SswapTilingData*>(tilingGm);
    tiling_.totalN = tilingPtr->totalN;
    tiling_.perCoreN = tilingPtr->perCoreN;
    tiling_.remainder = tilingPtr->remainder;
    tiling_.tileSize = tilingPtr->tileSize;
}

__aicore__ inline void SswapAIV::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    // aligned portion for DataCopy (256B = 64 floats aligned)
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    // CopyIn: GM -> UB (MTE2) - DataCopy for aligned, DataCopyPad only for tail
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    LocalTensor<float> yLocal = inQueueY_.AllocTensor<float>();

    if (alignedCount > 0) {
        DataCopy(xLocal, xGM_[curOffset], alignedCount);
        DataCopy(yLocal, yGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(xLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
        DataCopyPad(yLocal[alignedCount], yGM_[curOffset + alignedCount], copyParams, padParams);
    }

    // EnQue provides implicit MTE2->MTE3 sync (no explicit FetchEventID needed)
    inQueueX_.EnQue<float>(xLocal);
    inQueueY_.EnQue<float>(yLocal);

    // cross CopyOut: UB -> GM (MTE3) - x data to y position, y data to x position
    LocalTensor<float> xIn = inQueueX_.DeQue<float>();
    LocalTensor<float> yIn = inQueueY_.DeQue<float>();

    if (alignedCount > 0) {
        DataCopy(yGM_[curOffset], xIn, alignedCount);
        DataCopy(xGM_[curOffset], yIn, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGM_[curOffset + alignedCount], xIn[alignedCount], copyParams);
        DataCopyPad(xGM_[curOffset + alignedCount], yIn[alignedCount], copyParams);
    }

    inQueueX_.FreeTensor(xIn);
    inQueueY_.FreeTensor(yIn);
}

__aicore__ inline void SswapAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        SingleIteration(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }

    if (tileTail > 0) {
        SingleIteration(curOffset, tileTail);
    }
}

extern "C" __global__ __aicore__ void sswap_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; // R3: TPipe created in kernel entry, not as class member
    SswapAIV op;
    op.Init(x, y, tilingGm, &pipe);
    op.Process();
}

void sswap_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    sswap_kernel<<<numBlocks, nullptr, stream>>>(x, y, nullptr, tilingGm);
}
