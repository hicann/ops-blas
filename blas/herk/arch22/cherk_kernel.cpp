/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cherk_kernel.cpp
 * \brief CHERK kernels for Atlas A2 training series (arch22).
 *
 *        trans='N': C = alpha*A*A^H + beta*C
 *        trans='C': C = alpha*A^H*A + beta*C
 *        alpha and beta are real; C is Hermitian and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Phases 0 and 1 are the shared implementations in
 *        common/helper/complex_blas3_arch22.h; only the combine stage below is
 *        operator-specific.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cherk_kernel.h"
#include "cherk_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements per Phase 2 inner iteration; a whole multiple of a 256B fp32
// repeat so the GatherMask repeat count is exact.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");
} // namespace

// ===========================================================================
//  Phase 0 / Phase 1: shared bodies
// ===========================================================================
extern "C" __global__ __aicore__ void cherk_split_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitBody(a, ar, ai, tiling);
}

// Phase 0 for the trans='N' path: two K-interleaved operands whose Gram products
// give the real and imaginary parts directly, with the imaginary part's two signs
// alternating inside the accumulator (see SplitInterleaveBody).
extern "C" __global__ __aicore__ void cherk_split_interleave_kernel(
    GM_ADDR a, GM_ADDR p, GM_ADDR q, CBlas3SplitConcatTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitInterleaveBody(a, p, q, tiling);
}

extern "C" __global__ __aicore__ void cherk_gemm_kernel_tl(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransLeft, CBLAS3_GEMM_TRANS_LEFT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void cherk_gemm_kernel_tr(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransRight, CBLAS3_GEMM_TRANS_RIGHT>(left, right, out, tiling);
}

// ===========================================================================
//  Phase 2: combine into Hermitian C
// ===========================================================================
namespace {
// Everything the combine chunk loop needs, so the loop body can live in its own
// function. TPipe is deliberately NOT a member: rule R3 requires it to stay a
// local of the kernel entry.
struct CherkCombineCtx {
    GlobalTensor<float> t1Gm;
    GlobalTensor<float> t2Gm;
    GlobalTensor<float> t3Gm;
    GlobalTensor<float> t4Gm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage; // interleaved C_old in / C_new out
    LocalTensor<uint32_t> idxU;
    LocalTensor<float> laneIdx;
    LocalTensor<uint8_t> mask;
    bool applyAlpha;
    bool applyBeta;
};

// The nine UB scratch buffers backing a CherkCombineCtx. Declared in the caller's
// frame so their lifetime spans every CombineOneChunk call: the LocalTensors handed
// to ctx below stay in use for the whole column loop, so the TBufs they come from
// must outlive this function. Same arrangement as the shared layer's
// MirrorTriBuffers.
struct CherkCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> aux0;
    TBuf<TPosition::VECCALC> aux1;
    TBuf<TPosition::VECCALC> lane;
    TBuf<TPosition::VECCALC> mask;
};

__aicore__ inline void InitCombineCtx(
    TPipe& pipe, CherkCombineBuffers& bufs, CherkCombineCtx& ctx, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c, const CherkCombineTilingData& tiling)
{
    // alpha = 0 or k = 0 makes the host skip the whole temp allocation, so t1..t4
    // come in null. The alpha term is not evaluated in that case either
    // (skipAlphaTerm gates it below), so leave those four tensors unbound instead
    // of handing a null address to SetGlobalBuffer. beta may still be anything,
    // and C is always bound.
    if (tiling.skipAlphaTerm == 0U) {
        ctx.t1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1));
        ctx.t2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2));
        ctx.t3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3));
        ctx.t4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t4));
    }
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux0, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux1, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.lane, chunk * sizeof(float));
    pipe.InitBuffer(bufs.mask, chunk / cblas3::BITS_PER_BYTE + cblas3::BYTENUM_REPEAT);

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();
    ctx.laneIdx = bufs.lane.Get<float>();
    ctx.mask = bufs.mask.Get<uint8_t>();

    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    LocalTensor<int32_t> aux0 = bufs.aux0.Get<int32_t>();
    LocalTensor<int32_t> aux1 = bufs.aux1.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, aux0, aux1, chunk, chunk);
    ctx.idxU = idx.ReinterpretCast<uint32_t>();

    // Lane index consumed by the diagonal mask. aux0 is free again at this point,
    // the interleave table having already been materialised into idx.
    CreateVecIndex(aux0, 0, chunk);
    PipeBarrier<PIPE_V>();
    Cast(ctx.laneIdx, aux0, RoundMode::CAST_NONE, chunk);
    PipeBarrier<PIPE_V>();

    ctx.applyAlpha = (tiling.skipAlphaTerm == 0U);
    ctx.applyBeta = (tiling.isBetaZero == 0U);
}

// Cr = alpha*(t1 + t2), Ci = alpha*(t3 - t4), left in ctx.cr / ctx.ci.
//
// With fusedTemps the two additions have already happened inside the Cube: t1
// holds the whole real part and t3 the whole imaginary part, so only alpha is
// left to apply. The imaginary scale is negated there because this stage reads
// the temps with a column-major stride, which transposes them, and the imaginary
// part is antisymmetric: reading Pi transposed yields -Pi.
__aicore__ inline void AccumulateAlphaTerm(
    CherkCombineCtx& ctx, const CherkCombineTilingData& tiling, uint64_t tempOffset, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    const bool fused = (tiling.fusedTemps != 0U);

    DataCopyPad(ctx.cr, ctx.t1Gm[tempOffset], realParams, padParams);
    DataCopyPad(ctx.ci, ctx.t3Gm[tempOffset], realParams, padParams);
    if (!fused) {
        DataCopyPad(ctx.tmpA, ctx.t2Gm[tempOffset], realParams, padParams);
        DataCopyPad(ctx.tmpB, ctx.t4Gm[tempOffset], realParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (!fused) {
        Add(ctx.cr, ctx.cr, ctx.tmpA, len);
        Sub(ctx.ci, ctx.ci, ctx.tmpB, len);
        PipeBarrier<PIPE_V>();
    }
    Muls(ctx.cr, ctx.cr, tiling.alphaVal, len);
    Muls(ctx.ci, ctx.ci, fused ? -tiling.alphaVal : tiling.alphaVal, len);
    PipeBarrier<PIPE_V>();
}

// Adds beta * C_old into ctx.cr / ctx.ci, de-interleaving C_old on the way in.
__aicore__ inline void AccumulateBetaTerm(
    CherkCombineCtx& ctx, const CherkCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
{
    uint64_t rsvdCnt = 0;
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.stage, ctx.cGm[cOffset], cParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    GatherMask(ctx.tmpA, ctx.stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ctx.tmpB, ctx.stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    Axpy(ctx.cr, ctx.tmpA, tiling.betaVal, static_cast<int32_t>(len));
    Axpy(ctx.ci, ctx.tmpB, tiling.betaVal, static_cast<int32_t>(len));
    PipeBarrier<PIPE_V>();
}

// Cr = alpha*(t1 + t2) + beta*Cr_old, Ci = alpha*(t3 - t4) + beta*Ci_old, for one
// contiguous run of `len` rows inside column `col`.
__aicore__ inline void CombineOneChunk(
    CherkCombineCtx& ctx, const CherkCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    const uint32_t alignedLen = cblas3::AlignUpU(len, cblas3::ELENUM_REPEAT_FP32);
    const uint64_t tempOffset = static_cast<uint64_t>(col) * tiling.tempLdc + row;
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * cblas3::COMPLEX_ELENUM;

    if (ctx.applyAlpha) {
        AccumulateAlphaTerm(ctx, tiling, tempOffset, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
        PipeBarrier<PIPE_V>();
    }

    if (ctx.applyBeta) {
        AccumulateBetaTerm(ctx, tiling, cOffset, len);
    }

    const int32_t diagLocal = static_cast<int32_t>(col) - static_cast<int32_t>(row);
    cblas3::ZeroDiagonalImag(ctx.ci, ctx.laneIdx, ctx.mask, alignedLen, diagLocal);

    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`:
    // MTE3_V covers the Gather that *writes* stage (WAR), MTE3_MTE2 covers the
    // DataCopyPad that reads C_old into it. Relying on MTE2_V transitivity would
    // break on the path where neither alpha nor beta is applied, because that
    // iteration issues no MTE2 instruction at all.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace

extern "C" __global__ __aicore__ void cherk_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, CherkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const bool isUpper = (tiling.uploMode == CBLAS3_UPLO_UPPER);

    TPipe pipe;
    CherkCombineBuffers bufs;
    CherkCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, t1, t2, t3, t4, c, tiling);

    // Columns of a triangle carry very unequal work, so they are dealt out in a
    // round robin: every core then gets a uniform sample of the run lengths.
    for (uint32_t col = coreIdx; col < tiling.n; col += coreNum) {
        const uint32_t lo = isUpper ? 0U : col;
        const uint32_t hi = isUpper ? col : (tiling.n - 1U);
        const uint32_t runLen = hi - lo + 1U;

        for (uint32_t off = 0; off < runLen; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > runLen) ? (runLen - off) : COMBINE_CHUNK;
            CombineOneChunk(ctx, tiling, col, lo + off, len);
        }
    }
}

// ===========================================================================
//  Launchers
// ===========================================================================
void cherk_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    cherk_split_kernel<<<blockDim, nullptr, stream>>>(a, ar, ai, tiling);
}

void cherk_split_interleave_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR p, GM_ADDR q, CBlas3SplitConcatTilingData tiling)
{
    cherk_split_interleave_kernel<<<blockDim, nullptr, stream>>>(a, p, q, tiling);
}

void cherk_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    if (tiling.transMode == CBLAS3_GEMM_TRANS_LEFT) {
        cherk_gemm_kernel_tl<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    } else {
        cherk_gemm_kernel_tr<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    }
}

void cherk_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c,
    CherkCombineTilingData tiling)
{
    cherk_combine_kernel<<<blockDim, nullptr, stream>>>(t1, t2, t3, t4, c, tiling);
}

// Published so the host can split K without duplicating the compile-time
// singleCoreK of the Matmul static tiling.
uint32_t cherk_gemm_single_k() { return cblas3::SINGLE_K; }
