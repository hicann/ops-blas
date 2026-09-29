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
 * \file cher2k_kernel.cpp
 * \brief CHER2K kernels for Atlas A2 training series (arch22).
 *
 *        C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C   (trans = 'N')
 *        C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C   (trans = 'C')
 *        alpha is complex, beta is real; C is Hermitian and only the uplo triangle
 *        is referenced and updated. All matrices are column-major complex64.
 *
 *        Phases 0 and 1 are the shared implementations in
 *        common/helper/complex_blas3_arch22.h; only the combine stage below is
 *        operator-specific.
 *
 *        With M = A*B^H the whole expression collapses to
 *          C = alpha*M + conj(alpha)*M^H + beta*C
 *        because B*A^H = (A*B^H)^H = M^H. Phase 1 therefore computes M once, but
 *        in *both orientations*: A and B are distinct matrices, so unlike CHERK the
 *        fourth product is not the transpose of the third and no orientation comes
 *        for free. Eight GEMMs in total, four temps.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cher2k_kernel.h"
#include "cher2k_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements per Phase 2 inner iteration; a whole multiple of a 256B fp32
// repeat so the GatherMask repeat count is exact.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// TPipe is deliberately NOT a member: rule R3 requires it to stay a local of the
// kernel entry.
struct Cher2kCombineCtx {
    // Named by what a column-major read yields; see the header comment.
    GlobalTensor<float> mrGm;
    GlobalTensor<float> mrtGm;
    GlobalTensor<float> miGm;
    GlobalTensor<float> mitGm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> sumR; // Mr + Mr^T, then reused
    LocalTensor<float> difR; // Mr - Mr^T
    LocalTensor<float> sumI; // Mi + Mi^T
    LocalTensor<float> difI; // Mi - Mi^T
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage; // interleaved C_old in / C_new out
    LocalTensor<uint32_t> idxU;
    LocalTensor<float> laneIdx;
    LocalTensor<uint8_t> mask;
    bool applyAlpha;
    bool applyBeta;
};

// The thirteen UB scratch buffers backing a Cher2kCombineCtx. Declared in the
// caller's frame so their lifetime spans every CombineOneChunk call.
struct Cher2kCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> sumR;
    TBuf<TPosition::VECCALC> difR;
    TBuf<TPosition::VECCALC> sumI;
    TBuf<TPosition::VECCALC> difI;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> aux0;
    TBuf<TPosition::VECCALC> aux1;
    TBuf<TPosition::VECCALC> lane;
    TBuf<TPosition::VECCALC> mask;
};

__aicore__ inline void AllocCombineBuffers(TPipe& pipe, Cher2kCombineBuffers& bufs)
{
    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.sumR, chunk * sizeof(float));
    pipe.InitBuffer(bufs.difR, chunk * sizeof(float));
    pipe.InitBuffer(bufs.sumI, chunk * sizeof(float));
    pipe.InitBuffer(bufs.difI, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux0, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux1, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.lane, chunk * sizeof(float));
    pipe.InitBuffer(bufs.mask, chunk / cblas3::BITS_PER_BYTE + cblas3::BYTENUM_REPEAT);
}

__aicore__ inline void InitCombineCtx(
    TPipe& pipe, Cher2kCombineBuffers& bufs, Cher2kCombineCtx& ctx, GM_ADDR mrSrc, GM_ADDR mrtSrc, GM_ADDR miSrc,
    GM_ADDR mitSrc, GM_ADDR c, const Cher2kCombineTilingData& tiling)
{
    // alpha = (0,0) or k = 0 makes the host skip the whole temp allocation, so the
    // four M planes come in null. The alpha term is not evaluated in that case
    // either (skipAlphaTerm gates it below), so leave those tensors unbound instead
    // of handing a null address to SetGlobalBuffer. beta may still be anything, and
    // C is always bound.
    if (tiling.skipAlphaTerm == 0U) {
        ctx.mrGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mrSrc));
        ctx.mrtGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mrtSrc));
        ctx.miGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(miSrc));
        ctx.mitGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mitSrc));
    }
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t chunk = COMBINE_CHUNK;
    AllocCombineBuffers(pipe, bufs);

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.sumR = bufs.sumR.Get<float>();
    ctx.difR = bufs.difR.Get<float>();
    ctx.sumI = bufs.sumI.Get<float>();
    ctx.difI = bufs.difI.Get<float>();
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();
    ctx.laneIdx = bufs.lane.Get<float>();
    ctx.mask = bufs.mask.Get<uint8_t>();

    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    LocalTensor<int32_t> aux0 = bufs.aux0.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, aux0, bufs.aux1.Get<int32_t>(), chunk, chunk);
    ctx.idxU = idx.ReinterpretCast<uint32_t>();

    CreateVecIndex(aux0, 0, chunk);
    PipeBarrier<PIPE_V>();
    Cast(ctx.laneIdx, aux0, RoundMode::CAST_NONE, chunk);
    PipeBarrier<PIPE_V>();

    ctx.applyAlpha = (tiling.skipAlphaTerm == 0U);
    ctx.applyBeta = (tiling.isBetaZero == 0U);
}

// Cr = ar*(Mr + Mr^T) - ai*(Mi + Mi^T), Ci = ar*(Mi - Mi^T) + ai*(Mr - Mr^T).
__aicore__ inline void ApplyAlphaScale(
    Cher2kCombineCtx& ctx, const Cher2kCombineTilingData& tiling, uint64_t tempOffset, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};

    DataCopyPad(ctx.sumR, ctx.mrGm[tempOffset], realParams, padParams);  // Mr
    DataCopyPad(ctx.tmpA, ctx.mrtGm[tempOffset], realParams, padParams); // Mr^T
    // Both temps already carry their final sign: the subtraction of
    // Ar*Bi^T / Bi*Ar^T was folded into Phase 1 by negating Bi in place.
    DataCopyPad(ctx.sumI, ctx.miGm[tempOffset], realParams, padParams);  // Mi
    DataCopyPad(ctx.tmpB, ctx.mitGm[tempOffset], realParams, padParams); // Mi^T
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    Sub(ctx.difR, ctx.sumR, ctx.tmpA, len); // Mr - Mr^T
    Sub(ctx.difI, ctx.sumI, ctx.tmpB, len); // Mi - Mi^T
    PipeBarrier<PIPE_V>();
    Add(ctx.sumR, ctx.sumR, ctx.tmpA, len); // Mr + Mr^T
    Add(ctx.sumI, ctx.sumI, ctx.tmpB, len); // Mi + Mi^T
    PipeBarrier<PIPE_V>();

    Muls(ctx.cr, ctx.sumR, tiling.alphaRe, len);
    Muls(ctx.tmpA, ctx.sumI, tiling.alphaIm, len);
    PipeBarrier<PIPE_V>();
    Sub(ctx.cr, ctx.cr, ctx.tmpA, len);
    Muls(ctx.ci, ctx.difI, tiling.alphaRe, len);
    Muls(ctx.tmpB, ctx.difR, tiling.alphaIm, len);
    PipeBarrier<PIPE_V>();
    Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    PipeBarrier<PIPE_V>();
}

// Adds beta * C_old. beta is real for CHER2K, so this is a plain axpy per part.
__aicore__ inline void ApplyBetaScale(
    Cher2kCombineCtx& ctx, const Cher2kCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
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

//   Cr = ar*(Mr + Mr^T) - ai*(Mi + Mi^T) + beta*Cr_old
//   Ci = ar*(Mi - Mi^T) + ai*(Mr - Mr^T) + beta*Ci_old
// over `len` elements of one contiguous run of rows inside column `col`.
__aicore__ inline void CombineOneChunk(
    Cher2kCombineCtx& ctx, const Cher2kCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    const uint32_t alignedLen = cblas3::AlignUpU(len, cblas3::ELENUM_REPEAT_FP32);
    const uint64_t tempOffset = static_cast<uint64_t>(col) * tiling.tempLdc + row;
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * cblas3::COMPLEX_ELENUM;

    if (ctx.applyAlpha) {
        ApplyAlphaScale(ctx, tiling, tempOffset, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
        PipeBarrier<PIPE_V>();
    }

    if (ctx.applyBeta) {
        ApplyBetaScale(ctx, tiling, cOffset, len);
    }

    // C is Hermitian: Mi - Mi^T vanishes on the diagonal in exact arithmetic but
    // not in fp32, so the diagonal imaginary part is forced to zero.
    const int32_t diagLocal = static_cast<int32_t>(col) - static_cast<int32_t>(row);
    cblas3::ZeroDiagonalImag(ctx.ci, ctx.laneIdx, ctx.mask, alignedLen, diagLocal);

    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace

// ===========================================================================
//  Kernel entries
// ===========================================================================
extern "C" __global__ __aicore__ void cher2k_split_kernel(
    GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitBody(src, re, im, tiling);
}

extern "C" __global__ __aicore__ void cher2k_negate_kernel(GM_ADDR buf, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::NegateBody(buf, count);
}

extern "C" __global__ __aicore__ void cher2k_gemm_kernel_tl(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransLeft, CBLAS3_GEMM_TRANS_LEFT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void cher2k_gemm_kernel_tr(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransRight, CBLAS3_GEMM_TRANS_RIGHT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void cher2k_combine_kernel(
    GM_ADDR mrSrc, GM_ADDR mrtSrc, GM_ADDR miSrc, GM_ADDR mitSrc, GM_ADDR c, Cher2kCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const bool isUpper = (tiling.uploMode == CBLAS3_UPLO_UPPER);

    TPipe pipe;
    Cher2kCombineBuffers bufs;
    Cher2kCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, mrSrc, mrtSrc, miSrc, mitSrc, c, tiling);

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
void cher2k_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling)
{
    cher2k_split_kernel<<<blockDim, nullptr, stream>>>(src, re, im, tiling);
}

void cher2k_negate_kernel_do(uint32_t blockDim, void* stream, GM_ADDR buf, uint32_t count)
{
    cher2k_negate_kernel<<<blockDim, nullptr, stream>>>(buf, count);
}

void cher2k_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    if (tiling.transMode == CBLAS3_GEMM_TRANS_LEFT) {
        cher2k_gemm_kernel_tl<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    } else {
        cher2k_gemm_kernel_tr<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    }
}

void cher2k_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR mrSrc, GM_ADDR mrtSrc, GM_ADDR miSrc, GM_ADDR mitSrc, GM_ADDR c,
    Cher2kCombineTilingData tiling)
{
    cher2k_combine_kernel<<<blockDim, nullptr, stream>>>(mrSrc, mrtSrc, miSrc, mitSrc, c, tiling);
}

uint32_t cher2k_gemm_single_k() { return cblas3::SINGLE_K; }
