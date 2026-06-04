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
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "gemv_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BYTENUM_BLOCK       = 32;
constexpr uint32_t ELENUM_BLOCK_FP16    = 16;
constexpr uint32_t ELENUM_REPEAT_FP16   = 128;
// V pipe round-down granularity (= B32 / sizeof(float) = 8)
constexpr uint32_t VEC_FLOAT_PER_REPEAT = 32u / sizeof(float);

template <typename T, const bool TRANS_T = false, bool HSS_MODE = false>
class GemvBatchedAIV {
public:
    __aicore__ inline GemvBatchedAIV() {}
    __aicore__ inline void Init(
        GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();

    __aicore__ inline void InitUbuf();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void CopyInMatAndVec(uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr,
                                           uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyInY(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ApplyAlphaBeta(uint32_t curCalMatNum);
    __aicore__ inline void CastleHalfToFloat();
    __aicore__ inline void CastleYHalfToFloat();
    __aicore__ inline void CastFloatToHalf();
    __aicore__ inline void CopyOutRsltFloat(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyOutRsltHalf(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ComputeDotProduct(uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot);
    __aicore__ inline void OutputMChunkAndSync(uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign);

private:
    static constexpr bool IS_FLOAT     = IsSameType<T, float>::value;
    static constexpr bool IS_FLOAT_OUT = IS_FLOAT || HSS_MODE;    // y output is float

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<T> inxGM;
    GlobalTensor<T> inyGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> inyFloatGM;   // HSS: float y input
    GlobalTensor<float> outFloatGM;   // HSS: float y output
    GlobalTensor<float> workGM;

    TBuf<QuePosition::VECCALC> inABuf;
    TBuf<QuePosition::VECCALC> inxBuf;
    TBuf<QuePosition::VECCALC> inyBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> matTmpBuf;
    TBuf<QuePosition::VECCALC> vecTmpBuf;
    TBuf<QuePosition::VECCALC> matPreBuf;
    TBuf<QuePosition::VECCALC> vecPreBuf;

    LocalTensor<float> matTmpLocal;
    LocalTensor<float> vecTmpLocal;
    LocalTensor<half>   matPreLocal;
    LocalTensor<half>   vecPreLocal;
    LocalTensor<float> matFloatLocal;
    LocalTensor<float> vecFloatLocal;
    LocalTensor<float> yInFloatLocal;
    LocalTensor<float> mulResultLocal;
    LocalTensor<float> sumResultLocal;

    float alpha = 1.0f;
    float beta  = 0.0f;

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t outDim = 0;       // 输出向量维度
    uint32_t dotDim = 0;       // 点积向量维度
    uint32_t outTile = 0;        // outTile — 输出维度切分
    uint32_t dotTile = 0;        // dotTile — 点积维度切分
    uint32_t mCurr = 0;
    uint32_t calMatNum  = 0;
    uint32_t startMatId = 0;
    uint32_t batchGroupSize = 0;  // 一次搬多个batch进UB时使用
    uint32_t maxMatEleNum = 0;
    uint32_t maxVecEleNum = 0;
    // Host-precomputed buffer sizes (256B-aligned), read in ParseTilingData
    uint32_t szInA=0, szInx=0, szInY=0, szOut=0, szMatTmp=0, szVecTmp=0;
    int32_t lda = 0;
    int32_t vecIdx;
};

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::Init(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);
    this->inAGM.SetGlobalBuffer((__gm__ T *)A);
    this->inxGM.SetGlobalBuffer((__gm__ T *)x);
    if constexpr (HSS_MODE) {
        this->inyFloatGM.SetGlobalBuffer((__gm__ float *)y);
        this->outFloatGM.SetGlobalBuffer((__gm__ float *)y);
    } else {
        this->inyGM.SetGlobalBuffer((__gm__ T *)y);
        this->outGM.SetGlobalBuffer((__gm__ T *)y);
    }
    this->workGM.SetGlobalBuffer((__gm__ float *)workSpace);
    InitUbuf();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ParseTilingData(GM_ADDR tilingGm)
{
    const auto *td = reinterpret_cast<__gm__ GemvBatchedTilingData *>(tilingGm);
    this->alpha          = td->alpha;
    this->beta           = td->beta;
    this->m              = td->m;
    this->n              = td->n;
    this->outDim         = td->outSize;
    this->dotDim         = td->dotSize;
    this->batchGroupSize = td->batchGroupSize;  // 后续batchGroup搬入时使用
    this->dotTile          = td->dotTile;
    this->outTile          = td->outTile;
    this->szInA          = td->bufInA;
    this->szInx          = td->bufInx;
    this->szInY          = td->bufInY;
    this->szOut          = td->bufOut;
    this->szMatTmp       = td->bufMatTmp;
    this->szVecTmp       = td->bufVecTmp;
    this->lda            = td->lda;

    this->startMatId = this->vecIdx * td->batchPerCore;
    this->calMatNum  = (this->vecIdx == td->usedCoreNum - 1) ? td->batchTail : td->batchPerCore;

}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::InitUbuf()
{
    this->maxMatEleNum = this->outTile * this->dotTile;
    this->maxVecEleNum = (this->dotTile < 64u) ? 64u : this->dotTile;
    if constexpr (!TRANS_T) {
        if (this->outTile > this->maxVecEleNum) this->maxVecEleNum = this->outTile;
    }

    // Host-precomputed buffer sizes (256B-aligned)
    this->pipe.InitBuffer(this->inABuf,    this->szInA);
    this->pipe.InitBuffer(this->inxBuf,    this->szInx);
    this->pipe.InitBuffer(this->inyBuf,    this->szInY);
    this->pipe.InitBuffer(this->outBuf,    this->szOut);
    this->pipe.InitBuffer(this->matTmpBuf, this->szMatTmp);
    this->pipe.InitBuffer(this->vecTmpBuf, this->szVecTmp);

    if constexpr (!IS_FLOAT) {
        this->pipe.InitBuffer(this->matPreBuf, this->maxMatEleNum * sizeof(half));
        this->pipe.InitBuffer(this->vecPreBuf, this->maxVecEleNum * sizeof(half));
        this->matPreLocal = matPreBuf.Get<half>();
        this->vecPreLocal = vecPreBuf.Get<half>();
    }
    this->matTmpLocal = matTmpBuf.Get<float>();
    this->vecTmpLocal = vecTmpBuf.Get<float>();

    if constexpr (IS_FLOAT) {
        this->matFloatLocal  = this->matTmpLocal[0];
        this->mulResultLocal = this->matTmpLocal[0];  // same buffer for FP32
        this->sumResultLocal = this->vecTmpLocal[0];
        this->yInFloatLocal  = this->vecTmpLocal[this->outTile];
    } else {
        this->matFloatLocal  = this->matTmpLocal[0];
        this->mulResultLocal = this->matTmpLocal[this->maxMatEleNum];
        this->vecFloatLocal  = this->vecTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[this->dotTile];
        this->yInFloatLocal  = this->vecTmpLocal[this->dotTile + this->outTile];
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInMatAndVec(
    uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr,
    uint32_t mOffset, uint32_t mCurr)
{
    // nOffset/nCurr = output dim; mOffset/mCurr = inner dim
    // inA: loads nCurr "rows" each of mCurr elements
    //   Normal (outDim=m, inDim=n): m rows of n cols each
    //   Transpose (outDim=n, inDim=m): n cols of m rows each (column-major)
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    uint64_t inAOffset = (this->startMatId + curBatchId) * this->lda * this->n
                       + nOffset * (uint32_t)this->lda + mOffset;

    uint16_t matBlockCnt  = static_cast<uint16_t>(nCurr);
    uint32_t matBlockLen  = mCurr * sizeof(T);
    uint32_t matSrcStride = uint32_t(((uint32_t)this->lda - mCurr) * sizeof(T));
    uint32_t matDstStride = uint32_t((this->dotTile - mCurr) * sizeof(T) / BYTENUM_BLOCK);
    DataCopyExtParams copyMatParams{matBlockCnt, matBlockLen, matSrcStride, matDstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inALocal, this->inAGM[inAOffset], copyMatParams, padParams);

    uint64_t inxOffset = (this->startMatId + curBatchId) * this->dotDim + mOffset;
    DataCopyExtParams copyVecParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
    DataCopyPad(inxLocal, this->inxGM[inxOffset], copyVecParams, padParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInY(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    uint64_t offset = (this->startMatId + curBatchId) * this->outDim + mOffset;
    if constexpr (HSS_MODE) {
        auto inyLocal = inyBuf.Get<float>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyFloatGM[offset], copyParams, padParams);
    } else {
        LocalTensor<T> inyLocal = inyBuf.Get<T>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyGM[offset], copyParams, padParams);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ApplyAlphaBeta(uint32_t curCalMatNum)
{
    uint32_t count = curCalMatNum;
    if (alpha != 1.0f) Muls(this->sumResultLocal, this->sumResultLocal, alpha, count);
    if (beta != 0.0f) {
        Muls(this->mulResultLocal, this->yInFloatLocal, beta, count);
        PipeBarrier<PIPE_V>();
        Add(this->sumResultLocal, this->sumResultLocal, this->mulResultLocal, count);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastleYHalfToFloat()
{
    LocalTensor<half> inyLocal = inyBuf.Get<half>();
    DataCopy(this->vecPreLocal, inyLocal, this->mCurr);
    PipeBarrier<PIPE_V>();
    uint16_t castRepeatNum = (this->mCurr - 1) / ELENUM_REPEAT_FP16 + 1;
    Cast(this->yInFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP16);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastleHalfToFloat()
{
    LocalTensor<half> inALocal = inABuf.Get<half>();
    LocalTensor<half> inxLocal = inxBuf.Get<half>();

    uint32_t matRows = this->mCurr;
    uint16_t matCastRepeatNum = (matRows * this->dotTile - 1) / ELENUM_REPEAT_FP16 + 1;
    DataCopy(this->matPreLocal, inALocal, matRows * this->dotTile);

    DataCopy(this->vecPreLocal, inxLocal, this->dotTile);
    PipeBarrier<PIPE_V>();
    Cast(this->vecFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, this->dotTile);
    PipeBarrier<PIPE_V>();
    Cast(this->matFloatLocal, this->matPreLocal, RoundMode::CAST_NONE, matRows * this->dotTile);
    PipeBarrier<PIPE_V>();
}
template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastFloatToHalf()
{
    LocalTensor<half> outLocal = outBuf.Get<half>();
    uint16_t blockNum = (this->mCurr - 1) / ELENUM_BLOCK_FP16 + 1;
    Cast(outLocal, this->sumResultLocal, RoundMode::CAST_NONE, blockNum * ELENUM_BLOCK_FP16);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltFloat(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    uint64_t outOffset = (this->startMatId + curBatchId) * this->outDim + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
    if constexpr (HSS_MODE)
        DataCopyPad(this->outFloatGM[outOffset], this->sumResultLocal, copyParams);
    else
        DataCopyPad(this->outGM[outOffset], this->sumResultLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltHalf(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<half> outLocal = outBuf.Get<half>();
    uint64_t outOffset = (this->startMatId + curBatchId) * this->outDim + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(half)), 0, 0, 0};
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::Process()
{
    if (0 == this->calMatNum)
        return;

    uint32_t outTile = this->outTile;  // 输出维度切分
    uint32_t dotTile = this->dotTile;  // 点积维度切分
    uint32_t outDim = this->outDim;
    uint32_t dotDim = this->dotDim;

    for (uint32_t b = 0; b < this->calMatNum; b++) {
        for (uint32_t outOff = 0; outOff < outDim; outOff += outTile) {
            uint32_t outCnt = (outOff + outTile > outDim) ? (outDim - outOff) : outTile;
            uint32_t outCntAlign = ((outCnt + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));
            if (outCntAlign > outTile) outCntAlign = outTile;
            this->mCurr = outCnt;

            CopyInY(b, outOff, outCnt);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            bool isFirstDot = true;
            for (uint32_t dotOff = 0; dotOff < dotDim; dotOff += dotTile) {
                uint32_t dotCnt = (dotOff + dotTile > dotDim) ? (dotDim - dotOff) : dotTile;
                // CopyInMatAndVec: nOffset/nCurr=output; mOffset/mCurr=dot
                CopyInMatAndVec(b, outOff, outCnt, dotOff, dotCnt);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                if constexpr (!IS_FLOAT)
                    CastleHalfToFloat();

                ComputeDotProduct(outCnt, dotCnt, dotTile, isFirstDot);

                if (!isFirstDot)
                    Add(this->sumResultLocal, this->sumResultLocal, this->yInFloatLocal, outCntAlign);
                isFirstDot = false;
            }
            OutputMChunkAndSync(b, outOff, outCnt, outCntAlign);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ComputeDotProduct(
    uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot)
{
    constexpr uint32_t VL = 256 / sizeof(float);
    uint32_t lM = outCnt;
    uint32_t lNTile = dotTile;
    auto& dstLT = isFirstDot ? this->sumResultLocal : this->yInFloatLocal;
    __ubuf__ float* aAddr = IS_FLOAT ? (__ubuf__ float*)inABuf.Get<float>().GetPhyAddr()
                                     : (__ubuf__ float*)this->matFloatLocal.GetPhyAddr();
    __ubuf__ float* xAddr = IS_FLOAT ? (__ubuf__ float*)inxBuf.Get<float>().GetPhyAddr()
                                     : (__ubuf__ float*)this->vecFloatLocal.GetPhyAddr();
    __ubuf__ float* dAddr = (__ubuf__ float*)dstLT.GetPhyAddr();
    uint16_t vLoopNum = (dotCnt + VL - 1) / VL;
    uint32_t tailLen  = dotCnt % VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregA;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregX;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregMul;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregSum;
        auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(lM); row++) {
            AscendC::MicroAPI::Duplicate<float>(vregSum, 0.0f, maskAll);
            uint16_t matOff = row * static_cast<uint16_t>(lNTile);
            for (uint16_t i = 0; i < vLoopNum; i++) {
                uint32_t col = i * static_cast<uint32_t>(VL);
                uint32_t chunk = (i == vLoopNum - 1 && tailLen) ? tailLen : VL;
                auto mask = (chunk < VL)
                    ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk) : maskAll;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregA, (__ubuf__ float*)(aAddr + matOff + col));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregX, (__ubuf__ float*)(xAddr + col));
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregMul, vregA, vregX, mask);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregMul, maskAll);
            }
            AscendC::MicroAPI::ReduceSum(vregSum, vregSum, maskAll);
            __ubuf__ float* ptr = dAddr + row;
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, vregSum, 1, maskAll);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::OutputMChunkAndSync(
    uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign)
{
    PipeBarrier<PIPE_V>();
    if constexpr (IS_FLOAT_OUT) {
        DataCopy(this->yInFloatLocal, inyBuf.Get<float>(), mCurrAlign);
        PipeBarrier<PIPE_V>();
    } else {
        CastleYHalfToFloat();
    }
    ApplyAlphaBeta(mCurrAlign);
    if constexpr (!IS_FLOAT && !HSS_MODE)
        CastFloatToHalf();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if constexpr (IS_FLOAT_OUT)
        CopyOutRsltFloat(b, mOffset, mCurr);
    else
        CopyOutRsltHalf(b, mOffset, mCurr);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

// ============================================================
// SIMT — one thread per output element
// Column-major: A(row, col) = A[col * m + row]
// ============================================================
constexpr uint32_t GEMV_SIMT_MAX_THREADS = 2048;

template <typename T_IN, typename T_OUT, bool IS_TRANS>
__simt_vf__ __aicore__ LAUNCH_BOUND(GEMV_SIMT_MAX_THREADS)
inline void GemvSimt(uint32_t m, uint32_t n, uint32_t numB, uint32_t startB,
                      float alpha, float beta,
                      int32_t lda, int32_t incx, int32_t incy,
                      __gm__ const T_IN *aGm, __gm__ const T_IN *xGm, __gm__ T_OUT *yGm)
{
    constexpr bool IN_FP16  = IsSameType<T_IN, half>::value;
    constexpr bool OUT_FP16 = IsSameType<T_OUT, half>::value;
    uint32_t outDim = IS_TRANS ? n : m, dotDim = IS_TRANS ? m : n;
    uint32_t total = numB * outDim;
    uint32_t aBatch = (uint32_t)lda * n;
    uint32_t xBatch = 1u + (dotDim - 1u) * (uint32_t)abs(incx);
    uint32_t yBatch = 1u + (outDim - 1u) * (uint32_t)abs(incy);
    for (uint32_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        uint32_t b = tid / outDim, outIdx = tid % outDim;
        float acc = 0.0f;
        int32_t aOff = (int32_t)((startB + b) * aBatch);
        int32_t xOff = (int32_t)((startB + b) * xBatch);
        int32_t yOff = (int32_t)((startB + b) * yBatch);
        if (incx < 0) xOff += (int32_t)((dotDim - 1u) * (uint32_t)(-incx));
        if (incy < 0) yOff += (int32_t)((outDim - 1u) * (uint32_t)(-incy));
        for (uint32_t dotIdx = 0; dotIdx < dotDim; dotIdx++) {
            float av, xv;
            int32_t aIdx = IS_TRANS ? ((int32_t)outIdx * lda + (int32_t)dotIdx)
                                    : ((int32_t)dotIdx * lda + (int32_t)outIdx);
            int32_t xIdx = (int32_t)dotIdx * incx;
            if constexpr (IN_FP16) {
                av = __half2float(aGm[aOff + aIdx]);
                xv = __half2float(xGm[xOff + xIdx]);
            } else {
                av = (float)aGm[aOff + aIdx];
                xv = (float)xGm[xOff + xIdx];
            }
            acc += av * xv;
        }
        int32_t yIdx = (int32_t)outIdx * incy;
        float oldY;
        if constexpr (OUT_FP16)
            oldY = __half2float(yGm[yOff + yIdx]);
        else
            oldY = (float)yGm[yOff + yIdx];
        if constexpr (OUT_FP16)
            yGm[yOff + yIdx] = __float2half_rn_sat(alpha * acc + beta * oldY);
        else
            yGm[yOff + yIdx] = (T_OUT)(alpha * acc + beta * oldY);
    }
}

// ============================================================
// Kernel entry — SIMT for both Normal and Transpose (column-major)
// ============================================================
extern "C" __global__ __aicore__ void gemv_batched(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto *tdata = reinterpret_cast<__gm__ GemvBatchedTilingData *>(tilingGm);
    uint32_t startB = GetBlockIdx() * tdata->batchPerCore;
    uint32_t numB   = (GetBlockIdx() == tdata->usedCoreNum - 1) ? tdata->batchTail : tdata->batchPerCore;
    if (numB == 0) return;
    int32_t lda = tdata->lda, incx = tdata->incx, incy = tdata->incy;

    if (tdata->trans == 0) {  // Normal → SIMT
        if (tdata->dtype == 1) {       // FP32
            asc_vf_call<GemvSimt<float, float, false>>(dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta, lda, incx, incy,
                reinterpret_cast<__gm__ const float *>(A), reinterpret_cast<__gm__ const float *>(x),
                reinterpret_cast<__gm__ float *>(y));
        } else if (tdata->dtype == 0) { // FP16
            asc_vf_call<GemvSimt<half, half, false>>(dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta, lda, incx, incy,
                reinterpret_cast<__gm__ const half *>(A), reinterpret_cast<__gm__ const half *>(x),
                reinterpret_cast<__gm__ half *>(y));
        } else {                       // HSS
            asc_vf_call<GemvSimt<half, float, false>>(dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta, lda, incx, incy,
                reinterpret_cast<__gm__ const half *>(A), reinterpret_cast<__gm__ const half *>(x),
                reinterpret_cast<__gm__ float *>(y));
        }
    } else {  // Transpose: AIV (contiguous) or SIMT (strided)
        if (tdata->dtype == 2) {                       // HSS → AIV
            GemvBatchedAIV<half, true, true> op;
            op.Init(A, x, y, workSpace, tilingGm); op.Process();
        } else if (tdata->dtype == 1) {                // FP32
            if (tdata->incx == 1 && tdata->incy == 1) {
                GemvBatchedAIV<float> op; op.Init(A, x, y, workSpace, tilingGm); op.Process();
            } else {
                asc_vf_call<GemvSimt<float, float, true>>(dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                    tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta, lda, incx, incy,
                    reinterpret_cast<__gm__ const float *>(A), reinterpret_cast<__gm__ const float *>(x),
                    reinterpret_cast<__gm__ float *>(y));
            }
        } else {                                       // FP16
            if (tdata->incx == 1 && tdata->incy == 1) {
                GemvBatchedAIV<half> op; op.Init(A, x, y, workSpace, tilingGm); op.Process();
            } else {
                asc_vf_call<GemvSimt<half, half, true>>(dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                    tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta, lda, incx, incy,
                    reinterpret_cast<__gm__ const half *>(A), reinterpret_cast<__gm__ const half *>(x),
                    reinterpret_cast<__gm__ half *>(y));
            }
        }
    }
}

void gemv_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR y,
                            GM_ADDR workSpace, GM_ADDR tilingGm,
                            uint32_t numBlocks, void *stream)
{
    gemv_batched<<<numBlocks, nullptr, stream>>>(A, x, y, workSpace, tilingGm);
}
