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
#include "kernel_utils/kernel_constant.h"
#include "syr_tiling_data.h"

using namespace AscendC;

constexpr uint32_t UPLO_UPPER = 1;
constexpr uint32_t UPLO_LOWER = 0;

template <typename T>
class SyrKernel {
public:
    __aicore__ inline SyrKernel() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR A, GM_ADDR alpha,
                                GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;

    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void SyncMTEToV();
    __aicore__ inline void SyncVToMTE();

    // Full tile
    __aicore__ inline void CopyIn(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t row, uint32_t colOffset, uint32_t dataCount);

    // Head tile (UPPER mode, colBegin not tile-aligned)
    // Strategy: aligned read + full compute + offset write
    __aicore__ inline void CopyInHeadTile(uint32_t row, uint32_t alignedStart);
    __aicore__ inline void CopyOutHeadTile(uint32_t row, uint32_t colBegin,
                                           uint32_t offset, uint32_t dataCount);

    // Tail tile (remainder, DataCopyPad non-padding mode)
    __aicore__ inline void CopyInPad(uint32_t row, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputeImpl(uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t row, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void ProcessHeadTileBlock(uint32_t globalRow, uint32_t colBegin,
                                                  uint32_t colEnd, uint32_t &nextColStart);

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> aQueue;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> xGM;
    GlobalTensor<T> aGM;
    T alphaVal;

    int32_t blockIdx;
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    uint32_t useCoreNum;
    uint32_t rowStride;
    uint32_t rowStart;
    uint32_t tileSize;
    int32_t xRowOffset;   // offset of x[row] in first tile's xLocal, -1 if not available
    T scaledAlphaX;       // x[row] * alpha, precomputed once per row
};

template <typename T>
__aicore__ inline void SyrKernel<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ SyrTilingData *>(tilingGm);

    this->n = tiling->n;
    this->lda = tiling->lda;
    this->uplo = tiling->uplo;
    this->useCoreNum = tiling->useCoreNum;
    this->rowStride = tiling->rowStride;
    this->rowStart = tiling->startRow[this->blockIdx];
}

template <typename T>
__aicore__ inline void SyrKernel<T>::Init(TPipe* pipe, GM_ADDR x, GM_ADDR A, GM_ADDR alpha,
                                           GM_ADDR tilingGm)
{
    pipe_ = pipe;
    this->blockIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    // A is row-major with leading dimension lda: A[row][col] = A[row * lda + col]
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A), static_cast<uint32_t>(n * lda));
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(n));

    T alphaHost = *reinterpret_cast<__gm__ T*>(alpha);
    this->alphaVal = alphaHost;

    uint32_t ubSizePerBuffer = UB_SIZE / (BUFFER_NUM * 2 + 1);  // xQueue*2 + aQueue*2 + tmpBuf*1
    pipe_->InitBuffer(xQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe_->InitBuffer(aQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe_->InitBuffer(tmpBuf, ubSizePerBuffer);

    // Dynamically compute tile size: use full UB capacity to minimize GM round-trips.
    // Align to 32B boundary for optimal DMA transfer.
    uint32_t alignElements = 32 / sizeof(T);
    uint32_t maxElements = ubSizePerBuffer / sizeof(T);
    tileSize = (maxElements / alignElements) * alignElements;
}

template <typename T>
__aicore__ inline void SyrKernel<T>::SyncMTEToV()
{
    int32_t eventID = static_cast<int32_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventID);
}

template <typename T>
__aicore__ inline void SyrKernel<T>::SyncVToMTE()
{
    int32_t eventID = static_cast<int32_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
}

// ===== Full Tile Operations =====

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyIn(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    DataCopyExtParams cp{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyPad(xLocal, xGM[colOffset], cp, pp);
    xQueue.EnQue<T>(xLocal);

    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    DataCopyPad(aLocal, aGM[row * lda + colOffset], cp, pp);
    aQueue.EnQue<T>(aLocal);
}

template <typename T>
__aicore__ inline void SyrKernel<T>::ComputeImpl(uint32_t dataCount)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

    if (xRowOffset >= 0) {
        scaledAlphaX = xLocal.GetValue(static_cast<uint32_t>(xRowOffset)) * alphaVal;
        xRowOffset = -1;
    }

    Muls(tmpLocal, xLocal, scaledAlphaX, dataCount);
    Add(aLocal, aLocal, tmpLocal, dataCount);

    aQueue.EnQue<T>(aLocal);
    xQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyOut(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(aGM[row * lda + colOffset], aLocal, cp);
    aQueue.FreeTensor(aLocal);
}

// ===== Head Tile Operations (UPPER mode, colBegin not tile-aligned) =====

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyInHeadTile(uint32_t row, uint32_t alignedStart)
{
    DataCopyExtParams cp{1, static_cast<uint32_t>(tileSize * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyPad(xLocal, xGM[alignedStart], cp, pp);
    xQueue.EnQue<T>(xLocal);

    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    DataCopyPad(aLocal, aGM[row * lda + alignedStart], cp, pp);
    aQueue.EnQue<T>(aLocal);
}

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyOutHeadTile(uint32_t row, uint32_t colBegin,
                                                      uint32_t offset, uint32_t dataCount)
{
    // Fast path only: aligned read + offset write.
    // Caller guarantees:
    //   1. alignedStart + tileSize <= colEnd (read stays in bounds)
    //   2. (offset * sizeof(T)) % 32 == 0 (aLocal[offset] is 32B-aligned for DataCopyPad UB->GM)
    // When these conditions are not met, caller uses the DataCopyPad-based fallback instead.
    LocalTensor<T> aLocal = aQueue.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(aGM[row * lda + colBegin], aLocal[offset], copyParams);

    aQueue.FreeTensor(aLocal);
}

// ===== Tail Tile Operations (remainder, DataCopyPad non-padding mode) =====

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyInPad(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    // Non-padding mode: isPad=false, framework auto-fills dummy for 32B alignment
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyPad(xLocal, xGM[colOffset], copyParams, padParams);
    xQueue.EnQue<T>(xLocal);

    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    DataCopyPad(aLocal, aGM[row * lda + colOffset], copyParams, padParams);
    aQueue.EnQue<T>(aLocal);
}

template <typename T>
__aicore__ inline void SyrKernel<T>::CopyOutPad(uint32_t row, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> aLocal = aQueue.DeQue<T>();

    // DataCopyPad UB->GM: framework handles alignment, writes only dataCount elements to GM
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(aGM[row * lda + colOffset], aLocal, copyParams);

    aQueue.FreeTensor(aLocal);
}

// ===== Main Process =====

template <typename T>
__aicore__ inline void SyrKernel<T>::ProcessHeadTileBlock(uint32_t globalRow,
    uint32_t colBegin, uint32_t colEnd, uint32_t &nextColStart)
{
    if (uplo != UPLO_UPPER || (colBegin % tileSize == 0)) {
        nextColStart = colBegin;
        return;
    }

    uint32_t offset = colBegin % tileSize;
    uint32_t alignedStart = colBegin - offset;

    bool guardA = (alignedStart + tileSize <= colEnd);
    bool guardB = ((offset * sizeof(T)) % 32 == 0);

    if (guardA && guardB) {
        uint32_t dataCount = tileSize - offset;
        if (dataCount > colEnd - colBegin) {
            dataCount = colEnd - colBegin;
        }

        CopyInHeadTile(globalRow, alignedStart);
        SyncMTEToV();
        {
            LocalTensor<T> xLocal = xQueue.DeQue<T>();
            LocalTensor<T> aLocal = aQueue.DeQue<T>();
            LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
            if (xRowOffset >= 0) {
                scaledAlphaX = xLocal.GetValue(static_cast<uint32_t>(xRowOffset)) * alphaVal;
                xRowOffset = -1;
            }
            Muls(tmpLocal, xLocal[offset], scaledAlphaX, dataCount);
            Add(aLocal[offset], aLocal[offset], tmpLocal, dataCount);
            aQueue.EnQue<T>(aLocal);
            xQueue.FreeTensor(xLocal);
        }
        SyncVToMTE();
        CopyOutHeadTile(globalRow, colBegin, offset, dataCount);

        nextColStart = alignedStart + tileSize;
    } else {
        uint32_t nextAligned = alignedStart + tileSize;
        uint32_t leadCount = (nextAligned <= colEnd) ? (nextAligned - colBegin)
                                                      : (colEnd - colBegin);

        CopyInPad(globalRow, colBegin, leadCount);
        SyncMTEToV();
        xRowOffset = 0;  // CopyInPad reads from colBegin, x[row] at offset 0
        ComputeImpl(leadCount);
        SyncVToMTE();
        CopyOutPad(globalRow, colBegin, leadCount);

        nextColStart = colBegin + leadCount;
    }
}

template <typename T>
__aicore__ inline void SyrKernel<T>::Process()
{
    if (n == 0 || rowStart >= n) {
        return;
    }

    for (uint32_t globalRow = rowStart; globalRow < n; globalRow += rowStride) {
        uint32_t colBegin;
        uint32_t colEnd;
        if (uplo == UPLO_UPPER) {
            colBegin = globalRow;
            colEnd = n;
            // x[row] is always in the first x tile for UPPER mode
            xRowOffset = static_cast<int32_t>(globalRow % tileSize);
        } else {
            colBegin = 0;
            colEnd = globalRow + 1;
            // x[row] in first tile only if row < tileSize
            xRowOffset = (globalRow < tileSize) ? static_cast<int32_t>(globalRow) : -1;
        }

        // If x[row] not in first tile, read via DataCopyPad as fallback
        if (xRowOffset < 0) {
            LocalTensor<T> xTmp = tmpBuf.Get<T>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(xTmp, xGM[globalRow], cp, pp);
            SyncMTEToV();
            this->scaledAlphaX = xTmp.GetValue(0) * alphaVal;
        }

        uint32_t nextColStart;
        ProcessHeadTileBlock(globalRow, colBegin, colEnd, nextColStart);

        uint32_t remainingCols = (nextColStart < colEnd) ? (colEnd - nextColStart) : 0;
        uint32_t colTileNum = remainingCols / tileSize;
        uint32_t remainderCols = remainingCols % tileSize;

        for (uint32_t colTileIdx = 0; colTileIdx < colTileNum; ++colTileIdx) {
            uint32_t colOffset = nextColStart + colTileIdx * tileSize;
            CopyIn(globalRow, colOffset, tileSize);
            SyncMTEToV();
            ComputeImpl(tileSize);
            SyncVToMTE();
            CopyOut(globalRow, colOffset, tileSize);
        }

        if (remainderCols > 0) {
            uint32_t colOffset = nextColStart + colTileNum * tileSize;
            CopyInPad(globalRow, colOffset, remainderCols);
            SyncMTEToV();
            ComputeImpl(remainderCols);
            SyncVToMTE();
            CopyOutPad(globalRow, colOffset, remainderCols);
        }
    }
}

// ===== Kernel Entry (float32) =====

__global__ __aicore__ void syr_kernel(GM_ADDR x, GM_ADDR A, GM_ADDR alpha,
                                       GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SyrKernel<float> op;
    op.Init(&pipe, x, A, alpha, tilingGm);
    op.Process();
}

void syr_kernel_do(GM_ADDR x, GM_ADDR A, GM_ADDR alpha, GM_ADDR tilingGm,
                   uint32_t numBlocks, void* stream)
{
    syr_kernel<<<numBlocks, nullptr, stream>>>(x, A, alpha, tilingGm);
}


