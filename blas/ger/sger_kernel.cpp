/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SGER_KERNEL_H
#define SGER_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_UB_ELEMENTS = UB_SIZE / BYTENUM_PER_FLOAT32;
constexpr uint32_t MAX_CORE_NUM = 50;

struct SgerTilingDataDevice {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t startRow[MAX_CORE_NUM];
    uint32_t rowCount[MAX_CORE_NUM];
    uint32_t startCol[MAX_CORE_NUM];
    uint32_t colCount[MAX_CORE_NUM];
};

template <typename T>
class SgerKernel {
public:
    __aicore__ inline SgerKernel() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR alpha,
                                GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void CopyIn(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t row, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPad(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePad(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t row, uint32_t colOffset, uint32_t dataCount);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tmpQueue;

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    T alpha;

    int32_t blockIdx;
    int32_t blockNum;
    int64_t m;
    int64_t n;
    int64_t lda;

    uint32_t useCoreNum;
    uint32_t rowStart;
    uint32_t rowCount;
    uint32_t colStart;
    uint32_t colCount;
    uint32_t colTileNum;
    uint32_t remainderCols;
};

template <typename T>
__aicore__ inline void SgerKernel<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ SgerTilingDataDevice *>(tilingGm);

    this->m = tiling->m;
    this->n = tiling->n;
    this->lda = tiling->lda;
    this->useCoreNum = tiling->useCoreNum;

    this->rowStart = tiling->startRow[this->blockIdx];
    this->rowCount = tiling->rowCount[this->blockIdx];
    this->colStart = tiling->startCol[this->blockIdx];
    this->colCount = tiling->colCount[this->blockIdx];
}

template <typename T>
__aicore__ inline void SgerKernel<T>::Init(GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR alpha,
                                           GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->blockIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A), static_cast<uint32_t>(m * lda));
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(m));
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint32_t>(n));

    T alphaHost = *reinterpret_cast<__gm__ T*>(alpha);
    this->alpha = alphaHost;

    colTileNum = colCount / TILE_SIZE;
    remainderCols = colCount % TILE_SIZE;

    uint32_t ubSizePerBuffer = UB_SIZE / BUFFER_NUM / 3;
    pipe.InitBuffer(yQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe.InitBuffer(aQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, ubSizePerBuffer);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyIn(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    DataCopy(yLocal, yGM[colOffset], dataCount);
    yQueue.EnQue<T>(yLocal);

    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    uint32_t aRow = rowStart + row;
    DataCopy(aLocal, aGM[aRow * lda + colOffset], dataCount);
    aQueue.EnQue<T>(aLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::Compute(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();

    uint32_t aRow = rowStart + row;
    T xVal = xGM.GetValue(aRow);

    Muls(tmpLocal, yLocal, xVal, dataCount);
    Muls(tmpLocal, tmpLocal, alpha, dataCount);
    Add(aLocal, aLocal, tmpLocal, dataCount);

    tmpQueue.FreeTensor(tmpLocal);

    aQueue.EnQue<T>(aLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyOut(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    uint32_t aRow = rowStart + row;
    DataCopy(aGM[aRow * lda + colOffset], aLocal, dataCount);
    aQueue.FreeTensor(aLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyInPad(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = TILE_SIZE - dataCount;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    DataCopyPad(yLocal, yGM[colOffset], copyParams, padParams);
    yQueue.EnQue<T>(yLocal);

    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    uint32_t aRow = rowStart + row;
    DataCopyPad(aLocal, aGM[aRow * lda + colOffset], copyParams, padParams);
    aQueue.EnQue<T>(aLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::ComputePad(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();

    uint32_t aRow = rowStart + row;
    T xVal = xGM.GetValue(aRow);

    Muls(tmpLocal, yLocal, xVal, dataCount);
    Muls(tmpLocal, tmpLocal, alpha, dataCount);
    Add(aLocal, aLocal, tmpLocal, dataCount);

    tmpQueue.FreeTensor(tmpLocal);

    aQueue.EnQue<T>(aLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyOutPad(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = TILE_SIZE - dataCount;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    uint32_t aRow = rowStart + row;
    DataCopyPad(aGM[aRow * lda + colOffset], aLocal, copyParams);
    aQueue.FreeTensor(aLocal);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::Process()
{
    if (rowCount == 0 || colCount == 0) {
        return;
    }

    for (uint32_t row = 0; row < rowCount; ++row) {
        for (uint32_t colTileIdx = 0; colTileIdx < colTileNum; ++colTileIdx) {
            uint32_t colOffset = colStart + colTileIdx * TILE_SIZE;
            CopyIn(row, colOffset, TILE_SIZE);
            pipe_barrier(PIPE_ALL);
            Compute(row, colOffset, TILE_SIZE);
            pipe_barrier(PIPE_ALL);
            CopyOut(row, colOffset, TILE_SIZE);
        }

        if (remainderCols > 0) {
            uint32_t colOffset = colStart + colTileNum * TILE_SIZE;
            CopyInPad(row, colOffset, remainderCols);
            pipe_barrier(PIPE_ALL);
            ComputePad(row, colOffset, remainderCols);
            pipe_barrier(PIPE_ALL);
            CopyOutPad(row, colOffset, remainderCols);
        }
    }
}

__global__ __aicore__ void sger_kernel(GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR alpha,
                                       GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SgerKernel<float> op;
    op.Init(A, x, y, alpha, tilingGm);
    op.Process();
}

void sger_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR alpha, GM_ADDR tilingGm,
                   uint32_t numBlocks, void* stream)
{
    sger_kernel<<<numBlocks, nullptr, stream>>>(A, x, y, alpha, tilingGm);
}

#endif  // SGER_KERNEL_H
