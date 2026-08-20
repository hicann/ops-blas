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
#include "cann_ops_blas_common.h"
#include "chemm_tiling_data.h"
#include "chemm_kernel.h"

using namespace AscendC;

__aicore__ inline float ChemmAbs(float value)
{
    return value < 0.0f ? -value : value;
}

struct ChemmDoubleFloat {
    float high;
    float low;
};

__aicore__ inline ChemmDoubleFloat ChemmTwoSum(float left, float right)
{
    float high = left + right;
    float rightVirtual = high - left;
    float low = (left - (high - rightVirtual)) + (right - rightVirtual);
    return {high, low};
}

__aicore__ inline ChemmDoubleFloat ChemmTwoProduct(float left, float right)
{
    constexpr float splitter = 4097.0f;
    float high = left * right;
    float leftSplit = splitter * left;
    float leftHigh = leftSplit - (leftSplit - left);
    float leftLow = left - leftHigh;
    float rightSplit = splitter * right;
    float rightHigh = rightSplit - (rightSplit - right);
    float rightLow = right - rightHigh;
    float low = ((leftHigh * rightHigh - high) + leftHigh * rightLow + leftLow * rightHigh) +
        leftLow * rightLow;
    return {high, low};
}

__aicore__ inline ChemmDoubleFloat ChemmDfAdd(ChemmDoubleFloat left, ChemmDoubleFloat right)
{
    ChemmDoubleFloat highSum = ChemmTwoSum(left.high, right.high);
    ChemmDoubleFloat lowSum = ChemmTwoSum(left.low, right.low);
    ChemmDoubleFloat middle = ChemmTwoSum(highSum.low, lowSum.high);
    ChemmDoubleFloat result = ChemmTwoSum(highSum.high, middle.high);
    result.low += middle.low + lowSum.low;
    return ChemmTwoSum(result.high, result.low);
}

__aicore__ inline ChemmDoubleFloat ChemmDfNegate(ChemmDoubleFloat value)
{
    return {-value.high, -value.low};
}

__aicore__ inline ChemmDoubleFloat ChemmDfMulFloat(ChemmDoubleFloat value, float scalar)
{
    return ChemmDfAdd(ChemmTwoProduct(value.high, scalar), ChemmTwoProduct(value.low, scalar));
}

__aicore__ inline ChemmDoubleFloat ChemmDfLinearCombination(
    float left0, float right0, float left1, float right1)
{
    return ChemmDfAdd(ChemmTwoProduct(left0, right0), ChemmTwoProduct(left1, right1));
}

// ============================================================
// Workspace layout (all float, row-major ND):
//   A_r  [0,       m*k)       M×K
//   A_i  [m*k,     2*m*k)     M×K
//   B_r  [2*m*k,   2*m*k+k*n) K×N
//   B_i  [2*m*k+k*n, 2*m*k+2*k*n) K×N
//   RR   [2*m*k+2*k*n,                   ...+m*n)  M×N
//   II   [2*m*k+2*k*n+m*n,               ...+m*n)  M×N
//   RI   [2*m*k+2*k*n+2*m*n,             ...+m*n)  M×N
//   IR   [2*m*k+2*k*n+3*m*n,             ...+m*n)  M×N
// ============================================================

__aicore__ inline uint64_t WsOffsetAr(uint32_t m, uint32_t k) { return 0; }
__aicore__ inline uint64_t WsOffsetAi(uint32_t m, uint32_t k) { return static_cast<uint64_t>(m) * k; }
__aicore__ inline uint64_t WsOffsetBr(uint32_t m, uint32_t k, uint32_t n) { return 2ULL * m * k; }
__aicore__ inline uint64_t WsOffsetBi(uint32_t m, uint32_t k, uint32_t n) { return 2ULL * m * k + static_cast<uint64_t>(k) * n; }
__aicore__ inline uint64_t WsOffsetRR(uint32_t m, uint32_t k, uint32_t n) { return 2ULL * m * k + 2ULL * k * n; }
__aicore__ inline uint64_t WsOffsetII(uint32_t m, uint32_t k, uint32_t n) { return WsOffsetRR(m, k, n) + static_cast<uint64_t>(m) * n; }
__aicore__ inline uint64_t WsOffsetRI(uint32_t m, uint32_t k, uint32_t n) { return WsOffsetII(m, k, n) + static_cast<uint64_t>(m) * n; }
__aicore__ inline uint64_t WsOffsetIR(uint32_t m, uint32_t k, uint32_t n) { return WsOffsetRI(m, k, n) + static_cast<uint64_t>(m) * n; }

// ============================================================
// Phase 0: Preprocess (AIV, multi-core)
// Extract real/imag from complex interleaved A (Hermitian) and B.
// Expands Hermitian triangle to full matrix.
// Output: A_r, A_i (M×K), B_r, B_i (K×N) in workspace.
// ============================================================

__aicore__ inline void ChemmGetHermVal(
    GlobalTensor<float>& gm, uint32_t ld, bool isLower, uint32_t row, uint32_t col,
    float& outR, float& outI)
{
    if (row == col) {
        uint64_t idx = (static_cast<uint64_t>(row) * ld + col) * 2;
        outR = gm.GetValue(idx);
        outI = 0.0f;
    } else if ((isLower && row > col) || (!isLower && row < col)) {
        uint64_t idx = (static_cast<uint64_t>(row) * ld + col) * 2;
        outR = gm.GetValue(idx);
        outI = gm.GetValue(idx + 1);
    } else {
        uint64_t idx = (static_cast<uint64_t>(col) * ld + row) * 2;
        outR = gm.GetValue(idx);
        outI = -gm.GetValue(idx + 1);
    }
}

__aicore__ inline void ChemmStoreDoubleFloatResultWithOriginalC(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, float cReal, float cImag,
    uint32_t row, uint32_t column)
{
    bool betaIsZero = (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f);
    uint64_t cIdx = (static_cast<uint64_t>(row) * tiling.ldc + column) * 2U;
    ChemmDoubleFloat resultReal{0.0f, 0.0f};
    ChemmDoubleFloat resultImag{0.0f, 0.0f};
    if (!betaIsZero) {
        resultReal = ChemmDfLinearCombination(tiling.betaReal, cReal, -tiling.betaImag, cImag);
        resultImag = ChemmDfLinearCombination(tiling.betaReal, cImag, tiling.betaImag, cReal);
    }
    for (uint32_t kk = 0U; kk < tiling.k; ++kk) {
        float leftReal;
        float leftImag;
        float rightReal;
        float rightImag;
        if (isLeft) {
            ChemmGetHermVal(aG, tiling.lda, isLower, row, kk, leftReal, leftImag);
            uint64_t bIdx = (static_cast<uint64_t>(kk) * tiling.ldb + column) * 2U;
            rightReal = bG.GetValue(bIdx);
            rightImag = bG.GetValue(bIdx + 1U);
        } else {
            uint64_t bIdx = (static_cast<uint64_t>(row) * tiling.ldb + kk) * 2U;
            leftReal = bG.GetValue(bIdx);
            leftImag = bG.GetValue(bIdx + 1U);
            ChemmGetHermVal(aG, tiling.lda, isLower, kk, column, rightReal, rightImag);
        }
        ChemmDoubleFloat alphaLeftReal = ChemmDfLinearCombination(
            tiling.alphaReal, leftReal, -tiling.alphaImag, leftImag);
        ChemmDoubleFloat alphaLeftImag = ChemmDfLinearCombination(
            tiling.alphaReal, leftImag, tiling.alphaImag, leftReal);
        ChemmDoubleFloat realTerm = ChemmDfAdd(
            ChemmDfMulFloat(alphaLeftReal, rightReal),
            ChemmDfNegate(ChemmDfMulFloat(alphaLeftImag, rightImag)));
        ChemmDoubleFloat imagTerm = ChemmDfAdd(
            ChemmDfMulFloat(alphaLeftReal, rightImag),
            ChemmDfMulFloat(alphaLeftImag, rightReal));
        resultReal = ChemmDfAdd(resultReal, realTerm);
        resultImag = ChemmDfAdd(resultImag, imagTerm);
    }
    cG.SetValue(cIdx, resultReal.high + resultReal.low);
    cG.SetValue(cIdx + 1U, resultImag.high + resultImag.low);
}

__aicore__ inline void ChemmStoreDoubleFloatResult(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, uint32_t row, uint32_t column)
{
    bool betaIsZero = (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f);
    uint64_t cIdx = (static_cast<uint64_t>(row) * tiling.ldc + column) * 2U;
    float cReal = betaIsZero ? 0.0f : cG.GetValue(cIdx);
    float cImag = betaIsZero ? 0.0f : cG.GetValue(cIdx + 1U);
    ChemmStoreDoubleFloatResultWithOriginalC(
        aG, bG, cG, tiling, isLeft, isLower, cReal, cImag, row, column);
}

// Deinterleave aligned AoS B rows into the row-major fallback operands.
// Scalar handling preserves unaligned ldb prefixes and short tails.
__aicore__ inline void ChemmPackBRowToFallbackWorkspace(
    GlobalTensor<float>& bG, GlobalTensor<float>& ws, uint32_t ldb, uint32_t row, uint32_t columns,
    uint32_t stride, uint64_t realBase, uint64_t imagBase, LocalTensor<float>& aosLocal,
    LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal)
{
    constexpr uint32_t complexAlignment = 8U;
    constexpr uint32_t complexPerVector = 128U;
    event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    uint32_t column = 0U;
    while (column < columns) {
        uint64_t complexOffset = static_cast<uint64_t>(row) * ldb + column;
        uint32_t remaining = columns - column;
        uint64_t outputOffset = static_cast<uint64_t>(row) * stride + column;
        bool inputAligned = (complexOffset % complexAlignment) == 0U;
        bool realOutputAligned = ((realBase + outputOffset) % complexAlignment) == 0U;
        bool imagOutputAligned = ((imagBase + outputOffset) % complexAlignment) == 0U;
        if (!inputAligned || !realOutputAligned || !imagOutputAligned || remaining < complexAlignment) {
            uint64_t inputOffset = complexOffset * 2U;
            ws.SetValue(realBase + outputOffset, bG.GetValue(inputOffset));
            ws.SetValue(imagBase + outputOffset, bG.GetValue(inputOffset + 1U));
            ++column;
            continue;
        }
        uint32_t vectorCount = (remaining / complexAlignment) * complexAlignment;
        for (uint32_t offset = 0U; offset < vectorCount; offset += complexPerVector) {
            uint32_t currentCount = (vectorCount - offset) < complexPerVector ?
                (vectorCount - offset) : complexPerVector;
            DataCopy(aosLocal, bG[(complexOffset + offset) * 2U], currentCount * 2U);
            SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
            uint16_t repeatTimes = static_cast<uint16_t>((currentCount * 2U + 63U) / 64U);
            uint64_t reservedCount = 0U;
            GatherMask<float>(realLocal, aosLocal, 1U, false, 0U, {1U, repeatTimes, 8U, 8U}, reservedCount);
            GatherMask<float>(imagLocal, aosLocal, 2U, false, 0U, {1U, repeatTimes, 8U, 8U}, reservedCount);
            SetFlag<HardEvent::V_MTE3>(eventVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
            uint64_t vectorOutputOffset = outputOffset + offset;
            DataCopy(ws[realBase + vectorOutputOffset], realLocal, currentCount);
            DataCopy(ws[imagBase + vectorOutputOffset], imagLocal, currentCount);
            PipeBarrier<PIPE_ALL>();
        }
        column += vectorCount;
    }
}

__aicore__ inline void ChemmPackPreprocessRows(GlobalTensor<float>& bG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, uint32_t begin, uint32_t end, uint64_t arBase,
    uint64_t aiBase, uint64_t brBase, uint64_t biBase, LocalTensor<float>& aosLocal,
    LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal)
{
    uint32_t columns = isLeft ? tiling.n : tiling.k;
    uint32_t stride = columns;
    uint64_t realBase = isLeft ? brBase : arBase;
    uint64_t imagBase = isLeft ? biBase : aiBase;
    for (uint32_t row = begin; row < end; ++row) {
        ChemmPackBRowToFallbackWorkspace(bG, ws, tiling.ldb, row, columns, stride,
            realBase, imagBase, aosLocal, realLocal, imagLocal);
    }
}

__aicore__ inline void ChemmExpandPreprocessRows(GlobalTensor<float>& aG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, uint64_t begin, uint64_t end,
    uint64_t realBase, uint64_t imagBase)
{
    uint32_t columns = isLeft ? tiling.k : tiling.n;
    for (uint64_t index = begin; index < end; ++index) {
        uint32_t row = static_cast<uint32_t>(index / columns);
        uint32_t column = static_cast<uint32_t>(index % columns);
        float real;
        float imag;
        ChemmGetHermVal(aG, tiling.lda, isLower, row, column, real, imag);
        ws.SetValue(realBase + index, real);
        ws.SetValue(imagBase + index, imag);
    }
}

__aicore__ inline void ChemmInitPreprocessTensors(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspace,
    const ChemmMmadTiling& tiling, GlobalTensor<float>& aG, GlobalTensor<float>& bG,
    GlobalTensor<float>& ws)
{
    aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm),
        static_cast<uint64_t>(tiling.k) * tiling.lda * 2);
    bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm),
        static_cast<uint64_t>(tiling.m) * tiling.ldb * 2);
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
        WsOffsetIR(tiling.m, tiling.k, tiling.n) + static_cast<uint64_t>(tiling.m) * tiling.n);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
}

__aicore__ inline void ChemmRunPreprocessRows(GlobalTensor<float>& aG, GlobalTensor<float>& bG,
    GlobalTensor<float>& ws, const ChemmMmadTiling& tiling, bool isLeft, bool isLower,
    uint32_t blockIdx, uint32_t blockNum, uint64_t arBase, uint64_t aiBase, uint64_t brBase,
    uint64_t biBase, LocalTensor<float>& aosLocal, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal)
{
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    uint32_t bRows = isLeft ? tiling.k : tiling.m;
    uint32_t bBegin = static_cast<uint32_t>((static_cast<uint64_t>(blockIdx) * bRows) / blockNum);
    uint32_t bEnd = static_cast<uint32_t>((static_cast<uint64_t>(blockIdx + 1U) * bRows) / blockNum);
    ChemmPackPreprocessRows(bG, ws, tiling, isLeft, bBegin, bEnd, arBase, aiBase, brBase, biBase,
        aosLocal, realLocal, imagLocal);

    constexpr uint64_t alignment = 16U;
    uint64_t totalLeft = static_cast<uint64_t>(tiling.m) * tiling.k;
    uint64_t leftBegin = ((static_cast<uint64_t>(blockIdx) * totalLeft / blockNum + alignment - 1U) /
        alignment) * alignment;
    uint64_t leftEnd = ((static_cast<uint64_t>(blockIdx + 1U) * totalLeft / blockNum + alignment - 1U) /
        alignment) * alignment;
    if (leftEnd > totalLeft) leftEnd = totalLeft;
    if (isLeft) ChemmExpandPreprocessRows(aG, ws, tiling, true, isLower, leftBegin, leftEnd, arBase, aiBase);

    uint64_t totalRight = static_cast<uint64_t>(tiling.k) * tiling.n;
    uint64_t rightBegin = ((static_cast<uint64_t>(blockIdx) * totalRight / blockNum + alignment - 1U) /
        alignment) * alignment;
    uint64_t rightEnd = ((static_cast<uint64_t>(blockIdx + 1U) * totalRight / blockNum + alignment - 1U) /
        alignment) * alignment;
    if (rightEnd > totalRight) rightEnd = totalRight;
    if (!isLeft) ChemmExpandPreprocessRows(aG, ws, tiling, false, isLower, rightBegin, rightEnd, brBase, biBase);
}

extern "C" __global__ __aicore__ void chemm_preprocess_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspace, const ChemmMmadTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    bool isLeft = (tiling.sideMode == ACLBLAS_SIDE_LEFT);
    bool isLower = (tiling.uploMode == ACLBLAS_LOWER);

    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> ws;
    ChemmInitPreprocessTensors(aGm, bGm, workspace, tiling, aG, bG, ws);

    uint32_t bidx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bnum = static_cast<uint32_t>(GetBlockNum());
    if (bnum == 0) { bnum = 1; }

    uint64_t arBase = WsOffsetAr(tiling.m, tiling.k);
    uint64_t aiBase = WsOffsetAi(tiling.m, tiling.k);
    uint64_t brBase = WsOffsetBr(tiling.m, tiling.k, tiling.n);
    uint64_t biBase = WsOffsetBi(tiling.m, tiling.k, tiling.n);

    constexpr uint32_t fallbackChunkComplex = 128U;
    TPipe fallbackPipe;
    TBuf<TPosition::VECCALC> fallbackAosBuffer;
    TBuf<TPosition::VECCALC> fallbackRealBuffer;
    TBuf<TPosition::VECCALC> fallbackImagBuffer;
    fallbackPipe.InitBuffer(fallbackAosBuffer, fallbackChunkComplex * 2U * sizeof(float));
    fallbackPipe.InitBuffer(fallbackRealBuffer, fallbackChunkComplex * sizeof(float));
    fallbackPipe.InitBuffer(fallbackImagBuffer, fallbackChunkComplex * sizeof(float));
    LocalTensor<float> fallbackAosLocal = fallbackAosBuffer.Get<float>();
    LocalTensor<float> fallbackRealLocal = fallbackRealBuffer.Get<float>();
    LocalTensor<float> fallbackImagLocal = fallbackImagBuffer.Get<float>();
    ChemmRunPreprocessRows(aG, bG, ws, tiling, isLeft, isLower, bidx, bnum, arBase, aiBase, brBase, biBase,
        fallbackAosLocal, fallbackRealLocal, fallbackImagLocal);

    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    PipeBarrier<PIPE_ALL>();
}

// ============================================================
// Phase 1: GEMM
// Computes 4 real matrix multiplies:
//   RR = A_r * B_r,  II = A_i * B_i,  RI = A_r * B_i,  IR = A_i * B_r
// Each: C(i,j) = sum_k A(i,k) * B(k,j)
// ============================================================

constexpr uint32_t kChemmCubeBaseM = 64U;
constexpr uint32_t kChemmCubeBaseN = 64U;
constexpr uint32_t kChemmCubeBaseK = 32U;

// The Cube path consumes the row-major SoA matrices produced by preprocess.
// A single CO1 tile is reused only after its complete M -> Fixpipe lifecycle.
class ChemmBasicMmad {
public:
    __aicore__ inline void Init(
        __gm__ float* ar, __gm__ float* ai, __gm__ float* br, __gm__ float* bi,
        __gm__ float* rr, __gm__ float* ii, __gm__ float* ri, __gm__ float* ir,
        uint32_t m, uint32_t n, uint32_t k)
    {
        ar_.SetGlobalBuffer(ar, static_cast<uint64_t>(m) * k);
        ai_.SetGlobalBuffer(ai, static_cast<uint64_t>(m) * k);
        br_.SetGlobalBuffer(br, static_cast<uint64_t>(k) * n);
        bi_.SetGlobalBuffer(bi, static_cast<uint64_t>(k) * n);
        rr_.SetGlobalBuffer(rr, static_cast<uint64_t>(m) * n);
        ii_.SetGlobalBuffer(ii, static_cast<uint64_t>(m) * n);
        ri_.SetGlobalBuffer(ri, static_cast<uint64_t>(m) * n);
        ir_.SetGlobalBuffer(ir, static_cast<uint64_t>(m) * n);
        m_ = m;
        n_ = n;
        k_ = k;
    }

    __aicore__ inline void PrepareCo1()
    {
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void Process(uint32_t tileIndex, uint32_t tileStride)
    {
        // FP32 Cube format: c0=8, fractalNum=2, fractalSize=16*8.
        // For a 64x64x32 tile all A/B L1 and L0 allocations are 2048
        // floats, while the L0C allocation is 4096 floats.
        constexpr uint32_t aSizeAlignL1 = kChemmCubeBaseM * kChemmCubeBaseK;
        constexpr uint32_t bSizeAlignL1 = kChemmCubeBaseK * kChemmCubeBaseN;
        constexpr uint32_t aSizeAlignL0 = aSizeAlignL1;
        constexpr uint32_t bSizeAlignL0 = bSizeAlignL1;
        constexpr uint32_t cSizeAlignL0 = kChemmCubeBaseM * kChemmCubeBaseN;
        const uint32_t mTiles = m_ / kChemmCubeBaseM;
        const uint32_t nTiles = n_ / kChemmCubeBaseN;
        const uint64_t tileCount = static_cast<uint64_t>(mTiles) * nTiles;

        LocalTensor<float> a1(TPosition::A1, 0U, aSizeAlignL1);
        LocalTensor<float> b1(TPosition::B1, aSizeAlignL1 * sizeof(float), bSizeAlignL1);
        LocalTensor<float> a2(TPosition::A2, 0U, aSizeAlignL0);
        LocalTensor<float> b2(TPosition::B2, 0U, bSizeAlignL0);
        LocalTensor<float> c1(TPosition::CO1, 0U, cSizeAlignL0);

        for (uint64_t tile = tileIndex; tile < tileCount; tile += tileStride) {
            const uint32_t mBase = static_cast<uint32_t>((tile / nTiles) * kChemmCubeBaseM);
            const uint32_t nBase = static_cast<uint32_t>((tile % nTiles) * kChemmCubeBaseN);
            ProcessProduct(a1, b1, a2, b2, c1, ar_, br_, rr_, mBase, nBase);
            ProcessProduct(a1, b1, a2, b2, c1, ai_, bi_, ii_, mBase, nBase);
            ProcessProduct(a1, b1, a2, b2, c1, ar_, bi_, ri_, mBase, nBase);
            ProcessProduct(a1, b1, a2, b2, c1, ai_, br_, ir_, mBase, nBase);
        }
        // Drain the final Fixpipe completion token. Without this wait, a
        // later kernel invocation can block while reinitializing FIX_M.
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    __aicore__ inline void CopyToL1(
        LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
        uint32_t mBase, uint32_t nBase, uint32_t kBase)
    {
        Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = kChemmCubeBaseM;
        aParams.dValue = kChemmCubeBaseK;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = k_;
        aParams.dstNzC0Stride = kChemmCubeBaseM;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        DataCopy(a1, left[static_cast<uint64_t>(mBase) * k_ + kBase], aParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        Nd2NzParams bParams;
        bParams.ndNum = 1;
        bParams.nValue = kChemmCubeBaseK;
        bParams.dValue = kChemmCubeBaseN;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = n_;
        bParams.dstNzC0Stride = kChemmCubeBaseK;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;
        DataCopy(b1, right[static_cast<uint64_t>(kBase) * n_ + nBase], bParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void LoadToL0(
        LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2)
    {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        // FP32 NZ -> ZZ, equivalent to the official LoadData3DV2
        // false/false KernelMmad path. For 64x32: L1/L0=2048 floats.
        LoadData3DParamsV2<float> aParams;
        aParams.l1H = 1U;
        aParams.l1W = kChemmCubeBaseM;
        aParams.channelSize = kChemmCubeBaseK;
        aParams.kExtension = kChemmCubeBaseK;
        aParams.mExtension = kChemmCubeBaseM;
        aParams.strideW = 1U;
        aParams.strideH = 1U;
        aParams.filterW = 1U;
        aParams.filterH = 1U;
        aParams.dilationFilterW = 1U;
        aParams.dilationFilterH = 1U;
        aParams.filterSizeW = false;
        aParams.filterSizeH = false;
        aParams.enTranspose = false;
        aParams.fMatrixCtrl = false;
        LoadData(a2, a1, aParams);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        // FP32 NZ -> ZN. For K=32/N=64 this uses two K fractals and
        // eight N c0 repeats; the hardware transpose supplies ZN layout.
        LoadData3DParamsV2<float> bParams;
        bParams.l1H = 1U;
        bParams.l1W = kChemmCubeBaseK;
        bParams.channelSize = kChemmCubeBaseN;
        bParams.kExtension = kChemmCubeBaseN;
        bParams.mExtension = kChemmCubeBaseK;
        bParams.strideW = 1U;
        bParams.strideH = 1U;
        bParams.filterW = 1U;
        bParams.filterH = 1U;
        bParams.dilationFilterW = 1U;
        bParams.dilationFilterH = 1U;
        bParams.filterSizeW = false;
        bParams.filterSizeH = false;
        bParams.enTranspose = true;
        bParams.fMatrixCtrl = false;
        LoadData(b2, b1, bParams);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    }

    __aicore__ inline void Compute(
        LocalTensor<float>& c1, LocalTensor<float>& a2, LocalTensor<float>& b2, bool initialize)
    {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams params = {};
        params.m = kChemmCubeBaseM;
        params.n = kChemmCubeBaseN;
        params.k = kChemmCubeBaseK;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(c1, a2, b2, params);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessProduct(
        LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
        LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right,
        GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (uint32_t kBase = 0U; kBase < k_; kBase += kChemmCubeBaseK) {
            CopyToL1(a1, b1, left, right, mBase, nBase, kBase);
            LoadToL0(a1, b1, a2, b2);
            Compute(c1, a2, b2, kBase == 0U);
        }
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        StoreToGm(c1, output, mBase, nBase);
    }

    __aicore__ inline void StoreToGm(
        LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 params = {};
        params.mSize = kChemmCubeBaseM;
        params.nSize = kChemmCubeBaseN;
        params.srcStride = kChemmCubeBaseM;
        params.dstStride = n_;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    GlobalTensor<float> ar_;
    GlobalTensor<float> ai_;
    GlobalTensor<float> br_;
    GlobalTensor<float> bi_;
    GlobalTensor<float> rr_;
    GlobalTensor<float> ii_;
    GlobalTensor<float> ri_;
    GlobalTensor<float> ir_;
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
};

extern "C" __global__ __aicore__ void chemm_gemm_cube_kernel(
    GM_ADDR workspace, const ChemmMmadTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    __gm__ float* workspaceBase = reinterpret_cast<__gm__ float*>(workspace);
    ChemmBasicMmad mmad;
    mmad.PrepareCo1();
    uint64_t arOffset = WsOffsetAr(tiling.m, tiling.k);
    uint64_t aiOffset = WsOffsetAi(tiling.m, tiling.k);
    uint64_t brOffset = WsOffsetBr(tiling.m, tiling.k, tiling.n);
    uint64_t biOffset = WsOffsetBi(tiling.m, tiling.k, tiling.n);
    uint64_t rrOffset = WsOffsetRR(tiling.m, tiling.k, tiling.n);
    uint64_t iiOffset = WsOffsetII(tiling.m, tiling.k, tiling.n);
    uint64_t riOffset = WsOffsetRI(tiling.m, tiling.k, tiling.n);
    uint64_t irOffset = WsOffsetIR(tiling.m, tiling.k, tiling.n);
    mmad.Init(workspaceBase + arOffset, workspaceBase + aiOffset, workspaceBase + brOffset,
        workspaceBase + biOffset, workspaceBase + rrOffset, workspaceBase + iiOffset,
        workspaceBase + riOffset, workspaceBase + irOffset, tiling.m, tiling.n, tiling.k);
    mmad.Process(blockIdx, blockNum);

    GlobalTensor<float> workspaceTensor;
    workspaceTensor.SetGlobalBuffer(workspaceBase, WsOffsetIR(tiling.m, tiling.k, tiling.n) +
        static_cast<uint64_t>(tiling.m) * tiling.n);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(workspaceTensor);
    PipeBarrier<PIPE_ALL>();
}

struct ChemmFallbackAccumulator {
    float value;
    float correction;
};

__aicore__ inline void ChemmAccumulateFallback(float product, ChemmFallbackAccumulator& accumulator)
{
    float y = product - accumulator.correction;
    float t = accumulator.value + y;
    accumulator.correction = (t - accumulator.value) - y;
    accumulator.value = t;
}

__aicore__ inline void ChemmComputeFallbackElement(GlobalTensor<float>& ws, uint64_t arBase, uint64_t aiBase,
    uint64_t brBase, uint64_t biBase, uint64_t rrBase, uint64_t iiBase, uint64_t riBase, uint64_t irBase,
    uint32_t n, uint32_t k, uint64_t index)
{
    if (n == 0U) {
        return;
    }
    uint32_t row = static_cast<uint32_t>(index / n);
    uint32_t column = static_cast<uint32_t>(index % n);
    ChemmFallbackAccumulator rr{0.0f, 0.0f};
    ChemmFallbackAccumulator ii{0.0f, 0.0f};
    ChemmFallbackAccumulator ri{0.0f, 0.0f};
    ChemmFallbackAccumulator ir{0.0f, 0.0f};
    for (uint32_t kk = 0; kk < k; ++kk) {
        uint64_t aIndex = static_cast<uint64_t>(row) * k + kk;
        uint64_t bIndex = static_cast<uint64_t>(kk) * n + column;
        float ar = ws.GetValue(arBase + aIndex);
        float ai = ws.GetValue(aiBase + aIndex);
        float br = ws.GetValue(brBase + bIndex);
        float bi = ws.GetValue(biBase + bIndex);
        ChemmAccumulateFallback(ar * br, rr);
        ChemmAccumulateFallback(ai * bi, ii);
        ChemmAccumulateFallback(ar * bi, ri);
        ChemmAccumulateFallback(ai * br, ir);
    }
    ws.SetValue(rrBase + index, rr.value);
    ws.SetValue(iiBase + index, ii.value);
    ws.SetValue(riBase + index, ri.value);
    ws.SetValue(irBase + index, ir.value);
}

extern "C" __global__ __aicore__ void chemm_gemm_aiv_fallback_kernel(
    GM_ADDR workspace, const ChemmMmadTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t m = tiling.m;
    uint32_t n = tiling.n;
    uint32_t k = tiling.k;

    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                       WsOffsetIR(m, k, n) + static_cast<uint64_t>(m) * n);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);

    uint64_t arBase = WsOffsetAr(m, k);
    uint64_t aiBase = WsOffsetAi(m, k);
    uint64_t brBase = WsOffsetBr(m, k, n);
    uint64_t biBase = WsOffsetBi(m, k, n);
    uint64_t rrBase = WsOffsetRR(m, k, n);
    uint64_t iiBase = WsOffsetII(m, k, n);
    uint64_t riBase = WsOffsetRI(m, k, n);
    uint64_t irBase = WsOffsetIR(m, k, n);

    uint32_t bidx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bnum = static_cast<uint32_t>(GetBlockNum());
    if (bnum == 0U) {
        bnum = 1U;
    }

    uint64_t totalElems = static_cast<uint64_t>(m) * n;
    uint64_t elemsPerBlock = (totalElems + static_cast<uint64_t>(bnum) * 15U) /
        (static_cast<uint64_t>(bnum) * 16U) * 16U;
    uint64_t begin = static_cast<uint64_t>(bidx) * elemsPerBlock;
    uint64_t end = begin + elemsPerBlock;
    if (end > totalElems) {
        end = totalElems;
    }
    for (uint64_t idx = begin; idx < end; ++idx) {
        ChemmComputeFallbackElement(ws, arBase, aiBase, brBase, biBase, rrBase, iiBase, riBase, irBase, n, k, idx);
    }

    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    PipeBarrier<PIPE_ALL>();
}

// ============================================================
// Phase 2: Postprocess (AIV, multi-core)
// C = alpha * (RR-II, RI+IR) + beta * C
// ============================================================

__aicore__ inline void ChemmStoreFallbackEpilogueScalar(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, bool betaIsZero, uint64_t rrBase,
    uint64_t iiBase, uint64_t riBase, uint64_t irBase, uint64_t idx, uint32_t row, uint32_t column)
{
    constexpr float cancellationRatio = 0.0014f;
    float rr = ws.GetValue(rrBase + idx);
    float ii = ws.GetValue(iiBase + idx);
    float ri = ws.GetValue(riBase + idx);
    float ir = ws.GetValue(irBase + idx);
    float tmpReal = rr - ii;
    float tmpImag = ri + ir;
    uint64_t cIdx = (static_cast<uint64_t>(row) * tiling.ldc + column) * 2U;
    float cOrigR = betaIsZero ? 0.0f : cG.GetValue(cIdx);
    float cOrigI = betaIsZero ? 0.0f : cG.GetValue(cIdx + 1U);
    float termR0 = tiling.alphaReal * tmpReal;
    float termR1 = -tiling.alphaImag * tmpImag;
    float termR2 = tiling.betaReal * cOrigR;
    float termR3 = -tiling.betaImag * cOrigI;
    float termI0 = tiling.alphaReal * tmpImag;
    float termI1 = tiling.alphaImag * tmpReal;
    float termI2 = tiling.betaReal * cOrigI;
    float termI3 = tiling.betaImag * cOrigR;
    float outR = termR0 + termR1 + termR2 + termR3;
    float outI = termI0 + termI1 + termI2 + termI3;
    float scaleR = ChemmAbs(termR0) + ChemmAbs(termR1) + ChemmAbs(termR2) + ChemmAbs(termR3);
    float scaleI = ChemmAbs(termI0) + ChemmAbs(termI1) + ChemmAbs(termI2) + ChemmAbs(termI3);
    bool needsCorrection = (static_cast<uint64_t>(tiling.m) * tiling.n <= 64U) ||
        ChemmAbs(outR) <= cancellationRatio * scaleR || ChemmAbs(outI) <= cancellationRatio * scaleI;
    if (needsCorrection) {
        ChemmStoreDoubleFloatResult(aG, bG, cG, tiling, isLeft, isLower, row, column);
    } else {
        cG.SetValue(cIdx, outR);
        cG.SetValue(cIdx + 1U, outI);
    }
}

// The fast path retains original C in UB so correction can run after the
// vector result has been written to C.
__aicore__ inline void ChemmLoadVectorChunk(GlobalTensor<float>& cG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool betaIsZero, uint64_t rrBase, uint64_t iiBase, uint64_t riBase,
    uint64_t irBase, uint32_t row, uint32_t column, uint32_t count, LocalTensor<float>& rrLocal,
    LocalTensor<float>& iiLocal, LocalTensor<float>& riLocal, LocalTensor<float>& irLocal,
    LocalTensor<float>& cAosLocal, LocalTensor<float>& cRealLocal, LocalTensor<float>& cImagLocal)
{
    event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    uint64_t wsOffset = static_cast<uint64_t>(row) * tiling.n + column;
    uint64_t cOffset = (static_cast<uint64_t>(row) * tiling.ldc + column) * 2U;
    DataCopy(rrLocal, ws[rrBase + wsOffset], count);
    DataCopy(iiLocal, ws[iiBase + wsOffset], count);
    DataCopy(riLocal, ws[riBase + wsOffset], count);
    DataCopy(irLocal, ws[irBase + wsOffset], count);
    if (!betaIsZero) DataCopy(cAosLocal, cG[cOffset], count * 2U);
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
    if (betaIsZero) {
        Duplicate(cRealLocal, 0.0f, count);
        Duplicate(cImagLocal, 0.0f, count);
    } else {
        uint16_t repeatTimes = static_cast<uint16_t>((count * 2U + 63U) / 64U);
        uint64_t reservedCount = 0U;
        GatherMask<float>(cRealLocal, cAosLocal, 1U, false, 0U, {1U, repeatTimes, 8U, 8U}, reservedCount);
        GatherMask<float>(cImagLocal, cAosLocal, 2U, false, 0U, {1U, repeatTimes, 8U, 8U}, reservedCount);
    }
}

__aicore__ inline void ChemmComputeVectorChunk(const ChemmMmadTiling& tiling, uint32_t count,
    LocalTensor<float>& rrLocal, LocalTensor<float>& iiLocal, LocalTensor<float>& riLocal,
    LocalTensor<float>& irLocal, LocalTensor<float>& cRealLocal, LocalTensor<float>& cImagLocal,
    LocalTensor<float>& tmpRealLocal, LocalTensor<float>& tmpImagLocal, LocalTensor<float>& outRealLocal,
    LocalTensor<float>& outImagLocal, LocalTensor<float>& tempLocal)
{
    Sub(tmpRealLocal, rrLocal, iiLocal, count);
    Add(tmpImagLocal, riLocal, irLocal, count);
    Muls(outRealLocal, tmpRealLocal, tiling.alphaReal, count);
    Muls(tempLocal, tmpImagLocal, -tiling.alphaImag, count);
    Add(outRealLocal, outRealLocal, tempLocal, count);
    Muls(outImagLocal, tmpImagLocal, tiling.alphaReal, count);
    Muls(tempLocal, tmpRealLocal, tiling.alphaImag, count);
    Add(outImagLocal, outImagLocal, tempLocal, count);
    Muls(tempLocal, cRealLocal, tiling.betaReal, count);
    Add(outRealLocal, outRealLocal, tempLocal, count);
    Muls(tempLocal, cImagLocal, -tiling.betaImag, count);
    Add(outRealLocal, outRealLocal, tempLocal, count);
    Muls(tempLocal, cImagLocal, tiling.betaReal, count);
    Add(outImagLocal, outImagLocal, tempLocal, count);
    Muls(tempLocal, cRealLocal, tiling.betaImag, count);
    Add(outImagLocal, outImagLocal, tempLocal, count);
}

__aicore__ inline void ChemmStoreVectorChunk(GlobalTensor<float>& aG, GlobalTensor<float>& bG,
    GlobalTensor<float>& cG, const ChemmMmadTiling& tiling, bool isLeft, bool isLower, uint64_t cOffset,
    uint32_t row, uint32_t column, uint32_t count, LocalTensor<float>& cRealLocal, LocalTensor<float>& cImagLocal,
    LocalTensor<float>& tmpRealLocal, LocalTensor<float>& tmpImagLocal, LocalTensor<float>& outRealLocal,
    LocalTensor<float>& outImagLocal, LocalTensor<float>& outAosLocal)
{
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    for (uint32_t offset = 0U; offset < count; ++offset) {
        outAosLocal.SetValue(offset * 2U, outRealLocal.GetValue(offset));
        outAosLocal.SetValue(offset * 2U + 1U, outImagLocal.GetValue(offset));
    }
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    DataCopy(cG[cOffset], outAosLocal, count * 2U);
    SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
    for (uint32_t offset = 0U; offset < count; ++offset) {
        float tmpReal = tmpRealLocal.GetValue(offset);
        float tmpImag = tmpImagLocal.GetValue(offset);
        float cReal = cRealLocal.GetValue(offset);
        float cImag = cImagLocal.GetValue(offset);
        float scaleR = ChemmAbs(tiling.alphaReal * tmpReal) + ChemmAbs(tiling.alphaImag * tmpImag) +
            ChemmAbs(tiling.betaReal * cReal) + ChemmAbs(tiling.betaImag * cImag);
        float scaleI = ChemmAbs(tiling.alphaReal * tmpImag) + ChemmAbs(tiling.alphaImag * tmpReal) +
            ChemmAbs(tiling.betaReal * cImag) + ChemmAbs(tiling.betaImag * cReal);
        if (ChemmAbs(outRealLocal.GetValue(offset)) <= 0.0014f * scaleR ||
            ChemmAbs(outImagLocal.GetValue(offset)) <= 0.0014f * scaleI) {
            ChemmStoreDoubleFloatResultWithOriginalC(aG, bG, cG, tiling, isLeft, isLower,
                cReal, cImag, row, column + offset);
        }
    }
}

__aicore__ inline void ChemmProcessVectorChunk(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, bool betaIsZero, uint64_t rrBase,
    uint64_t iiBase, uint64_t riBase, uint64_t irBase, uint32_t row, uint32_t column, uint32_t count,
    LocalTensor<float>& rrLocal, LocalTensor<float>& iiLocal, LocalTensor<float>& riLocal,
    LocalTensor<float>& irLocal, LocalTensor<float>& cAosLocal, LocalTensor<float>& cRealLocal,
    LocalTensor<float>& cImagLocal, LocalTensor<float>& tmpRealLocal, LocalTensor<float>& tmpImagLocal,
    LocalTensor<float>& outRealLocal, LocalTensor<float>& outImagLocal, LocalTensor<float>& tempLocal,
    LocalTensor<float>& outAosLocal)
{
    uint64_t cOffset = (static_cast<uint64_t>(row) * tiling.ldc + column) * 2U;
    ChemmLoadVectorChunk(cG, ws, tiling, betaIsZero, rrBase, iiBase, riBase, irBase, row, column, count,
        rrLocal, iiLocal, riLocal, irLocal, cAosLocal, cRealLocal, cImagLocal);
    ChemmComputeVectorChunk(tiling, count, rrLocal, iiLocal, riLocal, irLocal, cRealLocal, cImagLocal,
        tmpRealLocal, tmpImagLocal, outRealLocal, outImagLocal, tempLocal);
    ChemmStoreVectorChunk(aG, bG, cG, tiling, isLeft, isLower, cOffset, row, column, count,
        cRealLocal, cImagLocal, tmpRealLocal, tmpImagLocal, outRealLocal, outImagLocal, outAosLocal);
}

__aicore__ inline void ChemmProcessVectorRows(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, bool betaIsZero, uint64_t rrBase,
    uint64_t iiBase, uint64_t riBase, uint64_t irBase, uint32_t rowBegin, uint32_t rowEnd,
    LocalTensor<float>& rrLocal, LocalTensor<float>& iiLocal, LocalTensor<float>& riLocal,
    LocalTensor<float>& irLocal, LocalTensor<float>& cAosLocal, LocalTensor<float>& cRealLocal,
    LocalTensor<float>& cImagLocal, LocalTensor<float>& tmpRealLocal, LocalTensor<float>& tmpImagLocal,
    LocalTensor<float>& outRealLocal, LocalTensor<float>& outImagLocal, LocalTensor<float>& tempLocal,
    LocalTensor<float>& outAosLocal)
{
    constexpr uint32_t chunkComplex = 128U;
    for (uint32_t row = rowBegin; row < rowEnd; ++row) {
        for (uint32_t column = 0U; column < tiling.n; column += chunkComplex) {
            uint32_t count = (tiling.n - column) < chunkComplex ? (tiling.n - column) : chunkComplex;
            ChemmProcessVectorChunk(aG, bG, cG, ws, tiling, isLeft, isLower, betaIsZero, rrBase, iiBase,
                riBase, irBase, row, column, count, rrLocal, iiLocal, riLocal, irLocal, cAosLocal,
                cRealLocal, cImagLocal, tmpRealLocal, tmpImagLocal, outRealLocal, outImagLocal,
                tempLocal, outAosLocal);
        }
    }
}

__aicore__ inline void ChemmInitVectorEpilogueBuffers(
    TPipe& pipe, TBuf<TPosition::VECCALC>& rrBuffer, TBuf<TPosition::VECCALC>& iiBuffer,
    TBuf<TPosition::VECCALC>& riBuffer, TBuf<TPosition::VECCALC>& irBuffer,
    TBuf<TPosition::VECCALC>& cAosBuffer, TBuf<TPosition::VECCALC>& cRealBuffer,
    TBuf<TPosition::VECCALC>& cImagBuffer, TBuf<TPosition::VECCALC>& tmpRealBuffer,
    TBuf<TPosition::VECCALC>& tmpImagBuffer, TBuf<TPosition::VECCALC>& outRealBuffer,
    TBuf<TPosition::VECCALC>& outImagBuffer, TBuf<TPosition::VECCALC>& tempBuffer,
    TBuf<TPosition::VECCALC>& outAosBuffer, LocalTensor<float>& rrLocal, LocalTensor<float>& iiLocal,
    LocalTensor<float>& riLocal, LocalTensor<float>& irLocal, LocalTensor<float>& cAosLocal,
    LocalTensor<float>& cRealLocal, LocalTensor<float>& cImagLocal, LocalTensor<float>& tmpRealLocal,
    LocalTensor<float>& tmpImagLocal, LocalTensor<float>& outRealLocal, LocalTensor<float>& outImagLocal,
    LocalTensor<float>& tempLocal, LocalTensor<float>& outAosLocal)
{
    constexpr uint32_t chunkComplex = 128U;
    pipe.InitBuffer(rrBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(iiBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(riBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(irBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(cAosBuffer, chunkComplex * 2U * sizeof(float));
    pipe.InitBuffer(cRealBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(cImagBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(tmpRealBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(tmpImagBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(outRealBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(outImagBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(tempBuffer, chunkComplex * sizeof(float));
    pipe.InitBuffer(outAosBuffer, chunkComplex * 2U * sizeof(float));
    rrLocal = rrBuffer.Get<float>();
    iiLocal = iiBuffer.Get<float>();
    riLocal = riBuffer.Get<float>();
    irLocal = irBuffer.Get<float>();
    cAosLocal = cAosBuffer.Get<float>();
    cRealLocal = cRealBuffer.Get<float>();
    cImagLocal = cImagBuffer.Get<float>();
    tmpRealLocal = tmpRealBuffer.Get<float>();
    tmpImagLocal = tmpImagBuffer.Get<float>();
    outRealLocal = outRealBuffer.Get<float>();
    outImagLocal = outImagBuffer.Get<float>();
    tempLocal = tempBuffer.Get<float>();
    outAosLocal = outAosBuffer.Get<float>();
}

__aicore__ inline bool ChemmTryVectorEpilogueRows(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG, GlobalTensor<float>& ws,
    const ChemmMmadTiling& tiling, bool isLeft, bool isLower, bool betaIsZero, uint64_t rrBase,
    uint64_t iiBase, uint64_t riBase, uint64_t irBase, uint32_t rowBegin, uint32_t rowEnd)
{
    constexpr uint32_t complexAlignment = 8U;
    if ((tiling.n % complexAlignment) != 0U || (tiling.ldc % complexAlignment) != 0U ||
        static_cast<uint64_t>(tiling.m) * tiling.n <= 64U) {
        return false;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> rrBuffer;
    TBuf<TPosition::VECCALC> iiBuffer;
    TBuf<TPosition::VECCALC> riBuffer;
    TBuf<TPosition::VECCALC> irBuffer;
    TBuf<TPosition::VECCALC> cAosBuffer;
    TBuf<TPosition::VECCALC> cRealBuffer;
    TBuf<TPosition::VECCALC> cImagBuffer;
    TBuf<TPosition::VECCALC> tmpRealBuffer;
    TBuf<TPosition::VECCALC> tmpImagBuffer;
    TBuf<TPosition::VECCALC> outRealBuffer;
    TBuf<TPosition::VECCALC> outImagBuffer;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> outAosBuffer;
    LocalTensor<float> rrLocal;
    LocalTensor<float> iiLocal;
    LocalTensor<float> riLocal;
    LocalTensor<float> irLocal;
    LocalTensor<float> cAosLocal;
    LocalTensor<float> cRealLocal;
    LocalTensor<float> cImagLocal;
    LocalTensor<float> tmpRealLocal;
    LocalTensor<float> tmpImagLocal;
    LocalTensor<float> outRealLocal;
    LocalTensor<float> outImagLocal;
    LocalTensor<float> tempLocal;
    LocalTensor<float> outAosLocal;
    ChemmInitVectorEpilogueBuffers(pipe, rrBuffer, iiBuffer, riBuffer, irBuffer, cAosBuffer, cRealBuffer,
        cImagBuffer, tmpRealBuffer, tmpImagBuffer, outRealBuffer, outImagBuffer, tempBuffer, outAosBuffer,
        rrLocal, iiLocal, riLocal, irLocal, cAosLocal, cRealLocal, cImagLocal, tmpRealLocal, tmpImagLocal,
        outRealLocal, outImagLocal, tempLocal, outAosLocal);

    ChemmProcessVectorRows(aG, bG, cG, ws, tiling, isLeft, isLower, betaIsZero, rrBase, iiBase,
        riBase, irBase, rowBegin, rowEnd, rrLocal, iiLocal, riLocal, irLocal, cAosLocal, cRealLocal,
        cImagLocal, tmpRealLocal, tmpImagLocal, outRealLocal, outImagLocal, tempLocal, outAosLocal);
    return true;
}

__aicore__ inline void ChemmInitEpilogueTensors(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR workspace,
    const ChemmMmadTiling& tiling, GlobalTensor<float>& aG, GlobalTensor<float>& bG,
    GlobalTensor<float>& cG, GlobalTensor<float>& ws)
{
    aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm),
        static_cast<uint64_t>(tiling.k) * tiling.lda * 2U);
    bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm),
        static_cast<uint64_t>(tiling.m) * tiling.ldb * 2U);
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
        WsOffsetIR(tiling.m, tiling.k, tiling.n) + static_cast<uint64_t>(tiling.m) * tiling.n);
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm),
        static_cast<uint64_t>(tiling.m) * tiling.ldc * 2U);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
}

__aicore__ inline void ChemmRunScalarEpilogue(GlobalTensor<float>& aG, GlobalTensor<float>& bG,
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const ChemmMmadTiling& tiling, bool isLeft,
    bool isLower, bool betaIsZero, uint64_t rrBase, uint64_t iiBase, uint64_t riBase, uint64_t irBase,
    uint32_t blockIdx, uint32_t blockNum)
{
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    if (tiling.n == 0U) return;
    constexpr uint64_t alignment = 8U;
    uint64_t totalElems = static_cast<uint64_t>(tiling.m) * tiling.n;
    uint64_t elemsPerBlock = (totalElems + static_cast<uint64_t>(blockNum) * alignment - 1U) /
        (static_cast<uint64_t>(blockNum) * alignment) * alignment;
    uint64_t begin = static_cast<uint64_t>(blockIdx) * elemsPerBlock;
    uint64_t end = begin + elemsPerBlock;
    if (end > totalElems) end = totalElems;
    for (uint64_t index = begin; index < end; ++index) {
        uint32_t row = static_cast<uint32_t>(index / tiling.n);
        uint32_t column = static_cast<uint32_t>(index % tiling.n);
        ChemmStoreFallbackEpilogueScalar(aG, bG, cG, ws, tiling, isLeft, isLower, betaIsZero,
            rrBase, iiBase, riBase, irBase, index, row, column);
    }
}

extern "C" __global__ __aicore__ void chemm_fallback_correction_epilogue_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR workspace, const ChemmMmadTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> ws;
    GlobalTensor<float> cG;
    ChemmInitEpilogueTensors(aGm, bGm, cGm, workspace, tiling, aG, bG, cG, ws);

    bool isLeft = (tiling.sideMode == ACLBLAS_SIDE_LEFT);
    bool isLower = (tiling.uploMode == ACLBLAS_LOWER);
    bool betaIsZero = (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f);
    uint64_t rrBase = WsOffsetRR(tiling.m, tiling.k, tiling.n);
    uint64_t iiBase = WsOffsetII(tiling.m, tiling.k, tiling.n);
    uint64_t riBase = WsOffsetRI(tiling.m, tiling.k, tiling.n);
    uint64_t irBase = WsOffsetIR(tiling.m, tiling.k, tiling.n);

    uint32_t bidx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bnum = static_cast<uint32_t>(GetBlockNum());
    if (bnum == 0) { bnum = 1; }

    uint32_t rowBegin = static_cast<uint32_t>((static_cast<uint64_t>(bidx) * tiling.m) / bnum);
    uint32_t rowEnd = static_cast<uint32_t>((static_cast<uint64_t>(bidx + 1U) * tiling.m) / bnum);
    if (ChemmTryVectorEpilogueRows(aG, bG, cG, ws, tiling, isLeft, isLower, betaIsZero, rrBase, iiBase,
        riBase, irBase, rowBegin, rowEnd)) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cG);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    ChemmRunScalarEpilogue(aG, bG, cG, ws, tiling, isLeft, isLower, betaIsZero,
        rrBase, iiBase, riBase, irBase, bidx, bnum);

    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cG);
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void chemm_beta_scale_kernel(
    GM_ADDR cGm, const ChemmMmadTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> cG;
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm),
        static_cast<uint64_t>(tiling.m) * tiling.ldc * 2U);
    bool betaIsZero = (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f);
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    constexpr uint64_t elementsPerCacheLine = 8U;
    uint64_t totalElems = static_cast<uint64_t>(tiling.m) * tiling.n;
    uint64_t elemsPerBlock = (totalElems + static_cast<uint64_t>(blockNum) * elementsPerCacheLine - 1U) /
        (static_cast<uint64_t>(blockNum) * elementsPerCacheLine) * elementsPerCacheLine;
    uint64_t begin = static_cast<uint64_t>(blockIdx) * elemsPerBlock;
    uint64_t end = begin + elemsPerBlock;
    if (end > totalElems) {
        end = totalElems;
    }
    for (uint64_t idx = begin; idx < end; ++idx) {
        uint32_t i = static_cast<uint32_t>(idx / tiling.n);
        uint32_t j = static_cast<uint32_t>(idx % tiling.n);
        uint64_t cIdx = (static_cast<uint64_t>(i) * tiling.ldc + j) * 2U;
        if (betaIsZero) {
            cG.SetValue(cIdx, 0.0f);
            cG.SetValue(cIdx + 1U, 0.0f);
        } else {
            float cReal = cG.GetValue(cIdx);
            float cImag = cG.GetValue(cIdx + 1U);
            cG.SetValue(cIdx, tiling.betaReal * cReal - tiling.betaImag * cImag);
            cG.SetValue(cIdx + 1U, tiling.betaReal * cImag + tiling.betaImag * cReal);
        }
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cG);
    PipeBarrier<PIPE_ALL>();
}

// ============================================================
// Host launcher
// ============================================================

void chemm_kernel_do(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR workspace,
    const ChemmMmadTiling& tiling, uint32_t aivCoreNum, void* stream)
{
    ChemmMmadTiling tilingCopy = tiling;
    tilingCopy.aivCoreNum = aivCoreNum;
    uint64_t outputElems = static_cast<uint64_t>(tilingCopy.m) * tilingCopy.n;
    uint32_t outputBlocks = static_cast<uint32_t>(outputElems < aivCoreNum ? outputElems : aivCoreNum);
    if (outputBlocks == 0U) {
        outputBlocks = 1U;
    }
    uint64_t preprocessElems = static_cast<uint64_t>(tilingCopy.m) * tilingCopy.k;
    uint64_t rightElems = static_cast<uint64_t>(tilingCopy.k) * tilingCopy.n;
    if (rightElems > preprocessElems) {
        preprocessElems = rightElems;
    }
    uint32_t preprocessBlocks = static_cast<uint32_t>(
        preprocessElems < aivCoreNum ? preprocessElems : aivCoreNum);
    if (tilingCopy.k < 16U || tilingCopy.n < 16U) {
        preprocessBlocks = 1U;
    }
    if (preprocessBlocks == 0U) {
        preprocessBlocks = 1U;
    }
    if (tilingCopy.alphaReal == 0.0f && tilingCopy.alphaImag == 0.0f) {
        if (tilingCopy.betaReal == 1.0f && tilingCopy.betaImag == 0.0f) {
            return;
        }
        chemm_beta_scale_kernel<<<outputBlocks, nullptr, stream>>>(cGm, tilingCopy);
        return;
    }
    chemm_preprocess_kernel<<<preprocessBlocks, nullptr, stream>>>(aGm, bGm, workspace, tilingCopy);
    // Each AIC owns a strided set of output tiles and serializes the four
    // complex-product components through one CO1 tile in a single launch.
    bool useCube = (tilingCopy.m % kChemmCubeBaseM) == 0U &&
        (tilingCopy.n % kChemmCubeBaseN) == 0U &&
        tilingCopy.k >= kChemmCubeBaseK && (tilingCopy.k % kChemmCubeBaseK) == 0U;
    if (useCube) {
        uint32_t cubeBlocks = tilingCopy.aicCoreNum;
        chemm_gemm_cube_kernel<<<cubeBlocks, nullptr, stream>>>(workspace, tilingCopy);
    } else {
        chemm_gemm_aiv_fallback_kernel<<<outputBlocks, nullptr, stream>>>(workspace, tilingCopy);
    }
    chemm_fallback_correction_epilogue_kernel<<<outputBlocks, nullptr, stream>>>(
        aGm, bGm, cGm, workspace, tilingCopy);
}
