/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef GBMV_KERNEL_H
#define GBMV_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "gbmv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

constexpr int32_t TRANS_N = 0;
constexpr int32_t TRANS_T = 1;
constexpr int32_t TRANS_C = 2;


template <typename T>
class GbmvAIV {
public:
    __aicore__ inline GbmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingGm, TPipe* pipe);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;

    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void ProcessScaleBeta();
    __aicore__ inline void ProcessTransN();
    __aicore__ inline void ProcessTransT();

    __aicore__ inline void SyncMTE3ToV();
    __aicore__ inline void SyncVtoMTE3();
    __aicore__ inline void CalcColSegment(uint32_t col, uint32_t &firstRow,
                                           uint32_t &lastRow, uint32_t &segLen);
    __aicore__ inline void ScaleBetaChunk(uint32_t &curOffset, uint32_t dataCount,
                                           bool usePad, uint8_t padNum);
    __aicore__ inline void LoadSegmentToQueue(TQue<QuePosition::VECIN, BUFFER_NUM> &q,
                                               GlobalTensor<T> &gm, uint64_t offset,
                                               uint32_t count, bool needPad, uint8_t padNum);
    __aicore__ inline void CopyOutSegmentPadAware(TQue<QuePosition::VECOUT, BUFFER_NUM> &q,
                                                    GlobalTensor<T> &gm, uint64_t offset,
                                                    uint32_t count, bool needPad, uint8_t padNum);
    __aicore__ inline T ProcessTransTBatch(uint64_t aOffset, uint64_t xOffset,
                                            uint32_t curCount, bool needPad, uint8_t padNum);

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    GlobalTensor<T> zGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TBuf<TPosition::VECCALC> tmpBuf;

    uint32_t vecIdx = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t kl = 0;
    uint32_t ku = 0;
    uint32_t lda = 0;
    uint32_t useCoreNum = 0;
    uint32_t maxSegLen = 0;
    int32_t trans = 0;

    uint32_t maxDataCount = 0;

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);

    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / sizeof(T);
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / sizeof(T);
};

// ===== Init / ParseTilingData =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingGm, TPipe* pipe)
{
    pipe_ = pipe;
    vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);

    aGM.SetGlobalBuffer((__gm__ T *)a, this->n * this->lda);

    if (trans == TRANS_N) {
        xGM.SetGlobalBuffer((__gm__ T *)x, this->n);
        yGM.SetGlobalBuffer((__gm__ T *)y, this->m);
        zGM.SetGlobalBuffer((__gm__ T *)z, this->m);
    } else {
        xGM.SetGlobalBuffer((__gm__ T *)x, this->m);
        yGM.SetGlobalBuffer((__gm__ T *)y, this->n);
        zGM.SetGlobalBuffer((__gm__ T *)z, this->n);
    }

    maxDataCount = 30 * 1024 / sizeof(T);

    if (trans == TRANS_N) {
        pipe_->InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
        pipe_->InitBuffer(xQueue, BUFFER_NUM, elementsPerBlock * sizeof(T));
        pipe_->InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));
        pipe_->InitBuffer(zQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    } else {
        pipe_->InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
        pipe_->InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
        pipe_->InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));
        pipe_->InitBuffer(zQueue, BUFFER_NUM, sizeof(T));

        int tmpCount = (maxDataCount / elementsPerRepeat + elementsPerBlock - 1) /
            elementsPerBlock * elementsPerBlock;
        pipe_->InitBuffer(tmpBuf, tmpCount * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void GbmvAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ GbmvTilingData<T> *>(tilingGm);

    m = tiling->m;
    n = tiling->n;
    kl = tiling->kl;
    ku = tiling->ku;
    lda = tiling->lda;
    useCoreNum = tiling->useCoreNum;
    trans = tiling->trans;
    alpha = tiling->alpha;
    beta = tiling->beta;
    maxSegLen = tiling->maxSegLen;

    if (useCoreNum == 0 || useCoreNum > MAX_CORE_NUM) {
        useCoreNum = 1;
    }
}

// ===== Sync Helpers =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::SyncMTE3ToV()
{
    int32_t eventID = static_cast<int32_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventID);
}

template <typename T>
__aicore__ inline void GbmvAIV<T>::SyncVtoMTE3()
{
    int32_t eventID = static_cast<int32_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
}

// ===== CalcColSegment =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::CalcColSegment(uint32_t col, uint32_t &firstRow,
                                                   uint32_t &lastRow, uint32_t &segLen)
{
    firstRow = (col > ku) ? (col - ku) : 0;
    lastRow = (col + kl < m) ? (col + kl) : (m - 1);
    segLen = (firstRow > lastRow) ? 0 : lastRow - firstRow + 1;
}

// ===== Load / CopyOut Helpers =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::LoadSegmentToQueue(TQue<QuePosition::VECIN, BUFFER_NUM> &q,
    GlobalTensor<T> &gm, uint64_t offset, uint32_t count, bool needPad, uint8_t padNum)
{
    if (!needPad) {
        LocalTensor<T> local = q.AllocTensor<T>();
        DataCopy(local, gm[offset], count);
        q.EnQue<T>(local);
    } else {
        DataCopyExtParams params{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, padNum, 0};
        LocalTensor<T> local = q.AllocTensor<T>();
        DataCopyPad(local, gm[offset], params, padParams);
        q.EnQue<T>(local);
    }
}

template <typename T>
__aicore__ inline void GbmvAIV<T>::CopyOutSegmentPadAware(TQue<QuePosition::VECOUT, BUFFER_NUM> &q,
    GlobalTensor<T> &gm, uint64_t offset, uint32_t count, bool needPad, uint8_t padNum)
{
    if (!needPad) {
        LocalTensor<T> local = q.DeQue<T>();
        DataCopy(gm[offset], local, count);
        q.FreeTensor(local);
    } else {
        DataCopyExtParams params{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, padNum, 0};
        LocalTensor<T> local = q.DeQue<T>();
        DataCopyPad(gm[offset], local, params);
        q.FreeTensor(local);
    }
}

// ===== ProcessScaleBeta =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::ScaleBetaChunk(uint32_t &curOffset, uint32_t dataCount,
                                                   bool usePad, uint8_t padNum)
{
    if (!usePad) {
        LocalTensor<T> yLocal = aQueue.AllocTensor<T>();
        DataCopy(yLocal, yGM[curOffset], dataCount);
        aQueue.EnQue<T>(yLocal);
    } else {
        DataCopyExtParams copyParamsIn{1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsIn{true, 0, padNum, 0};
        LocalTensor<T> yLocal = aQueue.AllocTensor<T>();
        DataCopyPad(yLocal, yGM[curOffset], copyParamsIn, padParamsIn);
        aQueue.EnQue<T>(yLocal);
    }

    SyncMTE3ToV();

    LocalTensor<T> yDeque = aQueue.DeQue<T>();
    LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
    Muls(zLocal, yDeque, beta, dataCount);

    SyncVtoMTE3();

    zQueue.EnQue<T>(zLocal);
    aQueue.FreeTensor(yDeque);

    if (!usePad) {
        LocalTensor<T> zDeque = zQueue.DeQue<T>();
        DataCopy(zGM[curOffset], zDeque, dataCount);
        zQueue.FreeTensor(zDeque);
    } else {
        DataCopyExtParams copyParamsOut{1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsOut{true, 0, padNum, 0};
        LocalTensor<T> zDeque = zQueue.DeQue<T>();
        DataCopyPad(zGM[curOffset], zDeque, copyParamsOut);
        zQueue.FreeTensor(zDeque);
    }

    curOffset += dataCount;
}

template <typename T>
__aicore__ inline void GbmvAIV<T>::ProcessScaleBeta()
{
    uint32_t yLen = (trans == TRANS_N) ? m : n;

    uint32_t repeatTimes = yLen / maxDataCount;
    uint32_t remainNum = yLen % maxDataCount;
    uint32_t curOffset = 0;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        ScaleBetaChunk(curOffset, maxDataCount, false, 0);
    }

    if (remainNum > 0) {
        uint8_t paddingNum = elementsPerBlock - remainNum % elementsPerBlock;
        if (paddingNum == elementsPerBlock) {
            paddingNum = 0;
        }
        ScaleBetaChunk(curOffset, remainNum, true, paddingNum);
    }
}

// ===== ProcessTransN =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::ProcessTransN()
{
    uint8_t paddingNumForScalar = elementsPerBlock - 1;

    for (uint32_t col = vecIdx; col < n; col += useCoreNum) {
        uint32_t firstRow, lastRow, segLen;
        CalcColSegment(col, firstRow, lastRow, segLen);
        if (segLen == 0) continue;

        uint32_t bandedStartRow = (ku > col) ? (ku - col) : 0;
        uint64_t aFlatIdx = bandedStartRow + static_cast<uint64_t>(col) * lda;
        bool aSegNeedPad = (segLen % elementsPerBlock != 0);
        uint8_t aPadNum = aSegNeedPad ? (elementsPerBlock - segLen % elementsPerBlock) : 0;

        LoadSegmentToQueue(aQueue, aGM, aFlatIdx, segLen, aSegNeedPad, aPadNum);

        // CopyIn x scalar
        {
            DataCopyExtParams copyParamsX{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParamsX{true, 0, paddingNumForScalar, 0};
            LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
            DataCopyPad(xLocal, xGM[col], copyParamsX, padParamsX);
            xQueue.EnQue<T>(xLocal);
        }

        SyncMTE3ToV();

        LocalTensor<T> aDeque = aQueue.DeQue<T>();
        LocalTensor<T> xDeque = xQueue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        T scalar = xDeque.GetValue(0);
        Muls(yLocal, aDeque, alpha * scalar, segLen);

        SyncVtoMTE3();

        yQueue.EnQue<T>(yLocal);
        aQueue.FreeTensor(aDeque);
        xQueue.FreeTensor(xDeque);

        CopyOutSegmentPadAware(yQueue, zGM, firstRow, segLen, aSegNeedPad, aPadNum);
    }
}

// ===== ProcessTransT =====

template <typename T>
__aicore__ inline T GbmvAIV<T>::ProcessTransTBatch(uint64_t aOffset, uint64_t xOffset,
                                                    uint32_t curCount, bool needPad, uint8_t padNum)
{
    LoadSegmentToQueue(aQueue, aGM, aOffset, curCount, needPad, padNum);
    LoadSegmentToQueue(xQueue, xGM, xOffset, curCount, needPad, padNum);

    SyncMTE3ToV();

    LocalTensor<T> aDeque = aQueue.DeQue<T>();
    LocalTensor<T> xDeque = xQueue.DeQue<T>();
    LocalTensor<T> workLocal = yQueue.AllocTensor<T>();
    Mul(workLocal, aDeque, xDeque, curCount);
    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
    ReduceSum(workLocal, workLocal, tmpLocal, curCount);

    SyncVtoMTE3();

    T curDot = workLocal.GetValue(0);
    yQueue.FreeTensor(workLocal);
    aQueue.FreeTensor(aDeque);
    xQueue.FreeTensor(xDeque);

    return curDot;
}

template <typename T>
__aicore__ inline void GbmvAIV<T>::ProcessTransT()
{
    uint8_t paddingNumForScalar = elementsPerBlock - 1;

    for (uint32_t col = vecIdx; col < n; col += useCoreNum) {
        uint32_t firstRow, lastRow, segLen;
        CalcColSegment(col, firstRow, lastRow, segLen);
        if (segLen == 0) continue;

        uint32_t bandedStartRow = (ku > col) ? (ku - col) : 0;
        uint64_t aFlatIdx = bandedStartRow + static_cast<uint64_t>(col) * lda;

        T dotAccum = static_cast<T>(0.0f);
        uint32_t segDone = 0;

        while (segDone < segLen) {
            uint32_t curCount = (segLen - segDone > maxDataCount) ? maxDataCount : (segLen - segDone);
            bool needPad = (curCount % elementsPerBlock != 0);
            uint8_t padNum = needPad ? (elementsPerBlock - curCount % elementsPerBlock) : 0;

            uint64_t aOffset = aFlatIdx + segDone;
            uint64_t xOffset = firstRow + segDone;

            dotAccum += ProcessTransTBatch(aOffset, xOffset, curCount, needPad, padNum);
            segDone += curCount;
        }

        dotAccum *= alpha;

        {
            LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
            zLocal.SetValue(0, dotAccum);
            zQueue.EnQue<T>(zLocal);

            LocalTensor<T> zDeque = zQueue.DeQue<T>();
            DataCopyExtParams copyParamsOut{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParamsOut{true, 0, paddingNumForScalar, 0};
            DataCopyPad(zGM[col], zDeque, copyParamsOut);
            zQueue.FreeTensor(zDeque);
        }
    }
}

// ===== Process =====

template <typename T>
__aicore__ inline void GbmvAIV<T>::Process()
{
    if (m == 0 || n == 0) {
        return;
    }

    SetAtomicAdd<T>();

    if (vecIdx == 0 && beta != static_cast<T>(0.0f)) {
        ProcessScaleBeta();
    }

    if (trans == TRANS_N) {
        ProcessTransN();
    } else {
        ProcessTransT();
    }

    SetAtomicNone();
}

// ===== Kernel Entry Points =====

__global__ __aicore__ void gbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workSpace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    TPipe pipe;
    GbmvAIV<float> op;
    op.Init(a, x, y, z, tilingGm, &pipe);
    op.Process();
}

void gbmv_kernel_do(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workSpace, GM_ADDR tilingGm,
    uint32_t numBlocks, void *stream)
{
    gbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, z, workSpace, tilingGm);
}

#endif  // GBMV_KERNEL_H
