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
constexpr uint32_t ELENUM_LINE_ALIGNED  = 32;
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
    __aicore__ inline void ComputeDotProduct(uint32_t mCurr, uint32_t nCurr, uint32_t nTile, bool isFirstNTile);
    __aicore__ inline void OutputMChunkAndSync(uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign);

private:
    static constexpr bool IS_FLOAT     = IsSameType<T, float>::value;
    static constexpr bool IS_FLOAT_OUT  = IS_FLOAT || HSS_MODE;    // y output is float
    static constexpr bool IS_TRANS = TRANS_T;
    static constexpr uint32_t BN = IS_TRANS ? 2 : 1;
    static constexpr uint32_t TN = IS_TRANS ? 2 : (IS_FLOAT ? 1 : 2);

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
    uint32_t nTile = 0;
    uint32_t mTile = 0;
    uint32_t mCurr = 0;
    uint32_t calMatNum  = 0;
    uint32_t startMatId = 0;
    uint32_t batchGroupSize = 0;
    uint32_t maxMatEleNum = 0;
    uint32_t maxVecEleNum = 0;
    // Host-precomputed buffer sizes (256B-aligned), read in ParseTilingData
    uint32_t szInA=0, szInx=0, szInY=0, szOut=0, szMatTmp=0, szVecTmp=0;
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
    this->batchGroupSize = td->batchGroupSize;
    this->nTile          = td->nTile;
    this->mTile          = td->mTile;
    this->szInA          = td->bufInA;
    this->szInx          = td->bufInx;
    this->szInY          = td->bufInY;
    this->szOut          = td->bufOut;
    this->szMatTmp       = td->bufMatTmp;
    this->szVecTmp       = td->bufVecTmp;

    this->startMatId = this->vecIdx * td->batchPerCore;
    this->calMatNum  = (this->vecIdx == td->usedCoreNum - 1) ? td->batchTail : td->batchPerCore;

    if constexpr (!IS_TRANS) {
        if (this->nTile == 0) this->nTile = ELENUM_LINE_ALIGNED;
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::InitUbuf()
{
    this->maxMatEleNum = this->mTile * this->nTile;
    this->maxVecEleNum = (this->nTile < 64u) ? 64u : this->nTile;
    if (this->mTile > this->maxVecEleNum) this->maxVecEleNum = this->mTile;

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
        this->yInFloatLocal  = this->vecTmpLocal[this->mTile];
    } else {
        this->matFloatLocal  = this->matTmpLocal[0];
        this->mulResultLocal = this->matTmpLocal[this->maxMatEleNum];
        this->vecFloatLocal  = this->vecTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[this->nTile];
        this->yInFloatLocal  = this->vecTmpLocal[this->nTile + this->mTile];
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInMatAndVec(
    uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr,
    uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    uint64_t inAOffset = (this->startMatId + curBatchId) * this->m * this->n
                       + mOffset * this->n + nOffset;

    uint16_t matBlockCnt  = static_cast<uint16_t>(mCurr);
    uint32_t matBlockLen  = nCurr * sizeof(T);
    uint32_t matSrcStride = uint32_t((this->n - nCurr) * sizeof(T));
    uint32_t matDstStride = uint32_t((this->nTile - nCurr) * sizeof(T) / BYTENUM_BLOCK);
    DataCopyExtParams copyMatParams{matBlockCnt, matBlockLen, matSrcStride, matDstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inALocal, this->inAGM[inAOffset], copyMatParams, padParams);

    uint64_t inxOffset = (this->startMatId + curBatchId) * this->n + nOffset;
    uint16_t vecBlockCnt = 1;
    uint32_t vecBlockLen = nCurr * sizeof(T);
    uint32_t vecSrcStride = uint32_t((this->n - nCurr) * sizeof(T));
    uint32_t vecDstStride = uint32_t((this->nTile - nCurr) * sizeof(T) / BYTENUM_BLOCK);
    DataCopyExtParams copyVecParams{vecBlockCnt, vecBlockLen, vecSrcStride, vecDstStride, 0};
    DataCopyPad(inxLocal, this->inxGM[inxOffset], copyVecParams, padParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInY(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    uint64_t offset = (this->startMatId + curBatchId) * this->m + mOffset;
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
    uint16_t matCastRepeatNum = (matRows * this->nTile - 1) / ELENUM_REPEAT_FP16 + 1;
    DataCopy(this->matPreLocal, inALocal, matRows * this->nTile);

    DataCopy(this->vecPreLocal, inxLocal, this->nTile);
    PipeBarrier<PIPE_V>();
    Cast(this->vecFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, this->nTile);
    PipeBarrier<PIPE_V>();
    Cast(this->matFloatLocal, this->matPreLocal, RoundMode::CAST_NONE, matRows * this->nTile);
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
    uint64_t outOffset = (this->startMatId + curBatchId) * this->m + mOffset;
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
    uint64_t outOffset = (this->startMatId + curBatchId) * this->m + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(half)), 0, 0, 0};
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::Process()
{
    if (0 == this->calMatNum)
        return;

    for (uint32_t b = 0; b < this->calMatNum; b++) {
        for (uint32_t mOffset = 0; mOffset < this->m; mOffset += this->mTile) {
            uint32_t mCurr = (mOffset + this->mTile > this->m) ? (this->m - mOffset) : this->mTile;
            uint32_t mCurrAlign = ((mCurr + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));
            if (mCurrAlign > this->mTile) mCurrAlign = this->mTile;
            this->mCurr = mCurr;

            CopyInY(b, mOffset, mCurr);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            bool isFirstNTile = true;
            for (uint32_t nOffset = 0; nOffset < this->n; nOffset += this->nTile) {
                uint32_t nCurr = (nOffset + this->nTile > this->n) ? (this->n - nOffset) : this->nTile;
                CopyInMatAndVec(b, nOffset, nCurr, mOffset, mCurr);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                if constexpr (!IS_FLOAT)
                    CastleHalfToFloat();

                ComputeDotProduct(mCurr, nCurr, this->nTile, isFirstNTile);

                if (!isFirstNTile)
                    Add(this->sumResultLocal, this->sumResultLocal, this->yInFloatLocal, mCurrAlign);
                isFirstNTile = false;
            }
            OutputMChunkAndSync(b, mOffset, mCurr, mCurrAlign);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ComputeDotProduct(
    uint32_t mCurr, uint32_t nCurr, uint32_t nTile, bool isFirstNTile)
{
    constexpr uint32_t VL = 256 / sizeof(float);
    uint32_t lM = mCurr;
    uint32_t lNTile = nTile;
    auto& dstLT = isFirstNTile ? this->sumResultLocal : this->yInFloatLocal;

    __ubuf__ float* aAddr;
    __ubuf__ float* xAddr;
    if constexpr (IS_FLOAT) {
        aAddr = (__ubuf__ float*)inABuf.Get<float>().GetPhyAddr();
        xAddr = (__ubuf__ float*)inxBuf.Get<float>().GetPhyAddr();
    } else {
        aAddr = (__ubuf__ float*)this->matFloatLocal.GetPhyAddr();
        xAddr = (__ubuf__ float*)this->vecFloatLocal.GetPhyAddr();
    }
    __ubuf__ float* dAddr = (__ubuf__ float*)dstLT.GetPhyAddr();

    uint16_t vLoopNum = (nCurr + VL - 1) / VL;
    uint32_t tailLen  = nCurr % VL;

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
                    ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk)
                    : maskAll;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregA, (__ubuf__ float*)(aAddr + matOff + col));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregX, (__ubuf__ float*)(xAddr + col));
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregMul, vregA, vregX, mask);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregSum, vregSum, vregMul, maskAll);
            }
            AscendC::MicroAPI::ReduceSum(vregSum, vregSum, maskAll);
            {
                __ubuf__ float* ptr = dAddr + row;
                AscendC::MicroAPI::DataCopy<
                    float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, vregSum, 1, maskAll);
            }
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
// SIMT trans — one thread per output element
// ============================================================
constexpr uint32_t GEMV_SIMT_MAX_THREADS = 2048;

template <bool IS_TRANS>
__simt_vf__ __aicore__ LAUNCH_BOUND(GEMV_SIMT_MAX_THREADS)
inline void GemvSimtTransFp32(uint32_t m, uint32_t n, uint32_t numB, uint32_t startB,
                               float alpha, float beta,
                               __gm__ const float *aGm, __gm__ const float *xGm, __gm__ float *yGm)
{
    uint32_t total = numB * n;
    for (uint32_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        uint32_t b = tid / n, k = tid % n;
        float acc  = 0.0f;
        uint32_t aOff = (startB + b) * m * n;
        uint32_t xOff = (startB + b) * m;
        for (uint32_t j = 0; j < m; j++) acc += aGm[aOff + j * n + k] * xGm[xOff + j];
        uint32_t yPos = (startB + b) * n + k;
        yGm[yPos] = alpha * acc + beta * yGm[yPos];
    }
}

template <bool IS_TRANS>
__simt_vf__ __aicore__ LAUNCH_BOUND(GEMV_SIMT_MAX_THREADS)
inline void GemvSimtTransFp16(uint32_t m, uint32_t n, uint32_t numB, uint32_t startB,
                               float alpha, float beta,
                               __gm__ const uint16_t *aGm, __gm__ const uint16_t *xGm, __gm__ uint16_t *yGm)
{
    uint32_t total = numB * n;
    for (uint32_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        uint32_t b = tid / n, k = tid % n;
        float acc  = 0.0f;
        uint32_t aOff = (startB + b) * m * n;
        uint32_t xOff = (startB + b) * m;
        for (uint32_t j = 0; j < m; j++)
            acc += __half2float(aGm[aOff + j * n + k]) * __half2float(xGm[xOff + j]);
        uint32_t yPos = (startB + b) * n + k;
        float yOld = __half2float(yGm[yPos]);
        yGm[yPos] = __float2half_rn_sat(alpha * acc + beta * yOld);
    }
}

// ============================================================
// Vec kernel entry — AIV for N, SIMT for T
// ============================================================
extern "C" __global__ __aicore__ void gemv_batched(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto *tdata = reinterpret_cast<__gm__ GemvBatchedTilingData *>(tilingGm);
    uint32_t coreIdx = GetBlockIdx();
    uint32_t startB = coreIdx * tdata->batchPerCore;
    uint32_t numB   = (coreIdx == tdata->usedCoreNum - 1) ? tdata->batchTail : tdata->batchPerCore;

    if (numB == 0) return;

    if (tdata->trans != 0) {
        if (tdata->dtype != 0) {
            asc_vf_call<GemvSimtTransFp32<true>>(
                dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta,
                reinterpret_cast<__gm__ const float *>(A),
                reinterpret_cast<__gm__ const float *>(x),
                reinterpret_cast<__gm__ float *>(y));
        } else {
            asc_vf_call<GemvSimtTransFp16<true>>(
                dim3{GEMV_SIMT_MAX_THREADS, 1, 1},
                tdata->m, tdata->n, numB, startB, tdata->alpha, tdata->beta,
                reinterpret_cast<__gm__ const uint16_t *>(A),
                reinterpret_cast<__gm__ const uint16_t *>(x),
                reinterpret_cast<__gm__ uint16_t *>(y));
        }
    } else if (tdata->dtype == 1) {
        GemvBatchedAIV<float> op;
        op.Init(A, x, y, workSpace, tilingGm);
        op.Process();
    } else if (tdata->dtype == 0) {
        GemvBatchedAIV<half> op;
        op.Init(A, x, y, workSpace, tilingGm);
        op.Process();
    } else {
        GemvBatchedAIV<half, false, true> op;
        op.Init(A, x, y, workSpace, tilingGm);
        op.Process();
    }
}

void gemv_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR y,
                            GM_ADDR workSpace, GM_ADDR tilingGm,
                            uint32_t numBlocks, void *stream)
{
    gemv_batched<<<numBlocks, nullptr, stream>>>(A, x, y, workSpace, tilingGm);
}
