/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPMV_KERNEL_H
#define TPMV_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t MAX_TILE_TASK = 8192;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
struct TpmvTilingDataDevice {
    uint32_t n;
    uint32_t useCoreNum;
    int64_t incx;
    uint32_t tileSize;
    uint32_t tileRows;
    uint32_t taskCount;
    uint16_t taskBi[MAX_TILE_TASK];
    uint32_t taskStart[MAX_CORE_NUM];
    uint32_t taskStep[MAX_CORE_NUM];
};

template <typename T>
class TpmvAIV {
public:
    __aicore__ inline TpmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline uint32_t PackedIndex(uint32_t i, uint32_t j) const;

    __aicore__ inline void CopyIn(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
    uint32_t useCoreNum = 0;
    uint32_t tileSize = 128;
    uint32_t tileRows = 0;
    uint32_t taskCount = 0;
    uint32_t taskStart = 0;
    uint32_t taskStep = 0;

    uint16_t taskBi[MAX_TILE_TASK] = {0};

    uint32_t maxDataCount = 0;

    int64_t incx = 1;

    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void TpmvAIV<T>::Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm)
{
    vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);

    xGM.SetGlobalBuffer((__gm__ T *)x, this->n);
    yGM.SetGlobalBuffer((__gm__ T *)y, this->n);
    aGM.SetGlobalBuffer((__gm__ T *)aPacked, this->n * this->n);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;  // 30kb / 4b

    // Workspace-related UB buffers are initialized here for later LocalTensor path enablement.
    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ TpmvTilingDataDevice<T> *>(tilingGm);

    n = tiling->n;
    useCoreNum = tiling->useCoreNum;
    incx = tiling->incx;
    tileSize = tiling->tileSize;
    tileRows = tiling->tileRows;
    taskCount = tiling->taskCount;

    if (tileSize == 0) {
        tileSize = 128;
    }
    if (useCoreNum == 0 || useCoreNum > MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    for (uint32_t i = 0; i < taskCount && i < MAX_TILE_TASK; ++i) {
        taskBi[i] = tiling->taskBi[i];
    }

    taskStart = tiling->taskStart[vecIdx];
    taskStep = tiling->taskStep[vecIdx];
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyIn(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    uint32_t r = colOffset + rowOffset * (rowOffset + 1) / 2;  // Start index of the row in packed storage.
    DataCopy(LocalA, aGM[r], dataCount);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::Compute(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalY, LocalA, LocalX, dataCount);
    ReduceSum(LocalY, LocalY, LocalY, dataCount);

    yQueue.EnQue<T>(LocalY);
    aQueue.FreeTensor(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyOut(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    DataCopyPad(yGM[rowOffset], yLocal, copyParams);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyInPad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint32_t r = colOffset + rowOffset * (rowOffset + 1) / 2;  // Start index of the row in packed storage.
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::ComputePad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalY, LocalA, LocalX, dataCount);
    ReduceSum(LocalY, LocalY, LocalY, dataCount);
    
    yQueue.EnQue<T>(LocalY);
    aQueue.FreeTensor(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyOutPad(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    DataCopyPad(yGM[rowOffset], yLocal, copyParams);
    yQueue.FreeTensor(yLocal);
}   

template <typename T>
__aicore__ inline void TpmvAIV<T>::Process()
{
    SetAtomicAdd<T>();

    uint32_t rowLen = n;
    uint32_t colLen = n;
    for (uint32_t col = 0; col < colLen; col += maxDataCount) {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        if (col + maxDataCount <= colLen) {
            DataCopy(xLocal, xGM[col], maxDataCount);
        } else {
            uint32_t dataCount = colLen - col;
            uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
            DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(xLocal, xGM[col], copyParams, padParams);
        }
        for (uint32_t taskIdx = taskStart; taskIdx < taskCount; taskIdx += taskStep) {
            uint32_t row = static_cast<uint32_t>(taskBi[taskIdx]);
            if (col > row) {
                continue;
            }
            uint32_t dataCount = maxDataCount;
            if (col + dataCount > row + 1) {
                dataCount = row + 1 - col;
                xQueue.EnQue<T>(xLocal);
                CopyInPad(taskIdx, row, col, dataCount);
                ComputePad(taskIdx, row, col, dataCount);
                CopyOutPad(taskIdx, row, col, dataCount);
                continue;
            }
            xQueue.EnQue<T>(xLocal);
            CopyIn(taskIdx, row, col, dataCount);
            Compute(taskIdx, row, col, dataCount);
            CopyOut(taskIdx, row, col, dataCount);
        }
        xQueue.FreeTensor(xLocal);
    }
    SetAtomicNone();
}

__global__ __aicore__ void tpmv_kernel(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TpmvAIV<float> op;
    op.Init(aPacked, x, y, tilingGm);
    op.Process();
}

void tpmv_kernel_do(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
    uint32_t numBlocks, void *stream)
{
    tpmv_kernel<<<numBlocks, nullptr, stream>>>(aPacked, x, y, workSpace, tilingGm);
}

#endif  // COPY_AIV_H
