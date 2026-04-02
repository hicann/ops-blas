/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYMV_KERNEL_H
#define SYMV_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t MAX_TILE_TASK = 4096;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
struct SymvTilingDataDevice {
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    T alpha;
    T beta;
    int64_t incx;
    int64_t incy;
    uint32_t tileSize;
    uint32_t tileRows;
    uint32_t taskCount;
    uint16_t taskBi[MAX_TILE_TASK];
    uint16_t taskBj[MAX_TILE_TASK];
    uint8_t taskType[MAX_TILE_TASK];
    uint32_t taskStart[MAX_CORE_NUM];
    uint32_t taskStep[MAX_CORE_NUM];
};

template <typename T>
class SymvAIV {
public:
    __aicore__ inline SymvAIV() = default;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void CopyInV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputeV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad);
    __aicore__ inline void ComputeVec(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad);

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    GlobalTensor<T> zGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workspaceQueue;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
    uint32_t lda = 0;
    uint32_t useCoreNum = 0;
    uint32_t tileSize = 128;
    uint32_t tileRows = 0;
    uint32_t taskCount = 0;
    uint32_t taskStart = 0;
    uint32_t taskStep = 0;

    uint16_t taskBi[MAX_TILE_TASK] = {0};
    uint16_t taskBj[MAX_TILE_TASK] = {0};

    uint32_t maxDataCount = 0;

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    int64_t incx = 1;
    int64_t incy = 1;
    
    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void SymvAIV<T>::Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingGm)
{
    vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);

    xGM.SetGlobalBuffer((__gm__ T *)x, this->n);
    yGM.SetGlobalBuffer((__gm__ T *)y, this->n);
    zGM.SetGlobalBuffer((__gm__ T *)z, this->n);
    aGM.SetGlobalBuffer((__gm__ T *)a, this->n * this->lda);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;  // 30kb / 4b

    // Workspace-related UB buffers are initialized here for later LocalTensor path enablement.
    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(x2Queue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(zQueue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(z2Queue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(workspaceQueue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ SymvTilingDataDevice<T> *>(tilingGm);

    n = tiling->n;
    lda = tiling->lda;
    useCoreNum = tiling->useCoreNum;
    alpha = tiling->alpha;
    beta = tiling->beta;
    incx = tiling->incx;
    incy = tiling->incy;
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
        taskBj[i] = tiling->taskBj[i];
    }

    taskStart = tiling->taskStart[vecIdx];
    taskStep = tiling->taskStep[vecIdx];
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint32_t r = rowOffset * lda + colOffset;  // Start index of the row in packed storage.
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopy(LocalA, aGM[r], dataCount);
    aQueue.EnQue<T>(LocalA);

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    DataCopy(LocalX, xGM[colOffset], dataCount);
    xQueue.EnQue<T>(LocalX);

    uint8_t paddingNum2 = elementsPerBlock - 1;
    DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};

    LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
    DataCopyPad(LocalX2, xGM[rowOffset], copyParams2, padParams2);
    x2Queue.EnQue<T>(LocalX2);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputeV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();
    LocalTensor<T> workLocal = workspaceQueue.AllocTensor<T>();

    // int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    // AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    // int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    // AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIDSToV);
    // AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIDSToV);
    if (colOffset + dataCount <= rowOffset) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }
    // int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    // AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    // AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    xQueue.FreeTensor(LocalX);
    x2Queue.FreeTensor(LocalX2);
    workspaceQueue.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    DataCopy(zGM[rowOffset], zLocal, dataCount);
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (colOffset + dataCount <= rowOffset) {
        uint8_t paddingNum2 = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        DataCopyPad(zGM[colOffset], z2Local, copyParams2);
    } else {
        uint8_t paddingNum2 = elementsPerBlock - (dataCount - 1) % elementsPerBlock;
        DataCopyExtParams copyParams2{1, (dataCount - 1) * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        DataCopyPad(zGM[colOffset], z2Local, copyParams2);
    }
    z2Queue.FreeTensor(z2Local);

}   

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint32_t r = rowOffset * lda + colOffset;  // Start index of the row in packed storage.
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    DataCopyPad(LocalX, xGM[colOffset], copyParams, padParams);
    xQueue.EnQue<T>(LocalX);

    uint8_t paddingNum2 = elementsPerBlock - 1;
    DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};

    LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
    DataCopyPad(LocalX2, xGM[rowOffset], copyParams2, padParams2);
    x2Queue.EnQue<T>(LocalX2);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputePadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();
    LocalTensor<T> workLocal = workspaceQueue.AllocTensor<T>();

    // int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    // AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    // int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    // AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIDSToV);
    // AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIDSToV);
    if (colOffset + dataCount <= rowOffset) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }
    // int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    // AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    // AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    xQueue.FreeTensor(LocalX);
    x2Queue.FreeTensor(LocalX2);
    workspaceQueue.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    DataCopyPad(zGM[rowOffset], zLocal, copyParams);
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (colOffset + dataCount <= rowOffset) {
        uint8_t paddingNum2 = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        DataCopyPad(zGM[colOffset], z2Local, copyParams2);
    } else {
        uint8_t paddingNum2 = elementsPerBlock - (dataCount - 1) % elementsPerBlock;
        DataCopyExtParams copyParams2{1, (dataCount - 1) * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        DataCopyPad(zGM[colOffset], z2Local, copyParams2);
    }
    z2Queue.FreeTensor(z2Local);

}   

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (!needPad) {
        LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
        DataCopy(LocalX, yGM[curOffset], dataCount);
        xQueue.EnQue<T>(LocalX);
        return;
    }

    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    DataCopyPad(LocalX, yGM[curOffset], copyParams, padParams);
    xQueue.EnQue<T>(LocalX);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputeVec(uint32_t curOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalZ = z2Queue.AllocTensor<T>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Muls(LocalZ, LocalX, beta, dataCount);
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    z2Queue.EnQue<T>(LocalZ);
    xQueue.FreeTensor(LocalX);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if(!needPad) {
        LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
        DataCopy(zGM[curOffset], LocalZ, dataCount);
        z2Queue.FreeTensor(LocalZ);
        return;
    }

    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
    DataCopyPad(zGM[curOffset], LocalZ, copyParams);
    z2Queue.FreeTensor(LocalZ);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::Process()
{    
    SetAtomicAdd<T>();
    if (vecIdx == 0) {
        uint32_t repeatTimes = n / maxDataCount;
        uint32_t remainNum = n % maxDataCount;
        uint32_t curOffset = 0;
        for (uint32_t i = 0; i < repeatTimes; i++) {
            uint32_t dataCount = maxDataCount;
            CopyInVec(curOffset, dataCount, false);
            ComputeVec(curOffset, dataCount);
            CopyOutVec(curOffset, dataCount, false);
            curOffset += dataCount;
        }
        if (remainNum > 0) {
            uint32_t dataCount = remainNum;
            CopyInVec(curOffset, dataCount, true);
            ComputeVec(curOffset, dataCount);
            CopyOutVec(curOffset, dataCount, true);
        }
    }

    int64_t incyStep = incy == 0 ? 1 : incy;
    for (uint32_t taskIdx = taskStart; taskIdx < taskCount; taskIdx += taskStep) {
        uint32_t bi = static_cast<uint32_t>(taskBi[taskIdx]);
        uint32_t bj = static_cast<uint32_t>(taskBj[taskIdx]);
        uint32_t row = bi;
        uint32_t colLen = bi + 1;
        for (uint32_t col = 0; col < colLen; col+= maxDataCount) {
            uint32_t dataCount = maxDataCount;
            if (col + dataCount > colLen) {
                dataCount = colLen - col;
                CopyInPadV2(taskIdx, row, col, dataCount);
                ComputePadV2(taskIdx, row, col, dataCount);
                CopyOutPadV2(taskIdx, row, col, dataCount);
                continue;
            }
            CopyInV2(taskIdx, row, col, dataCount);
            ComputeV2(taskIdx, row, col, dataCount);
            CopyOutV2(taskIdx, row, col, dataCount);
        }
    }
    SetAtomicNone();
}

__global__ __aicore__ void symv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workSpace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SymvAIV<float> op;
    op.Init(a, x, y, z, tilingGm);
    op.Process();
}

void symv_kernel_do(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workSpace, GM_ADDR tilingGm,
    uint32_t numBlocks, void *stream)
{
    symv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, z, workSpace, tilingGm);
}

#endif  // SYMV_KERNEL_H