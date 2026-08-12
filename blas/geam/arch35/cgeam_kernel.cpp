/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_kernel.cpp
 * @brief Cgeam kernel implementation (SIMD/membase, arch35)
 *        Computes: C = alpha * op(A) + beta * op(B) for complex matrices
 *
 *        Architecture: TPipe + TQue<VECIN>/TQue<VECOUT> + TBuf<VECCALC>
 *        EnQue/DeQue provides automatic MTE <-> Vector synchronization.
 *
 *        Complex representation:
 *          - Interleaved storage: r0,i0,r1,i1,...,rN,iN
 *          - Each complex matrix element = 2 adjacent floats (real, imag)
 *          - Separate real/imag buffers managed through individual TQue/TBuf
 *
 *        Buffer planning (each buffer holds tileM floats):
 *          - inQueueAR_, inQueueAI_ (VECIN): A tile real/imag, MTE2→V sync via TQue
 *          - calcBufBR_, calcBufBI_ (VECCALC): B tile real/imag when beta!=0; scratch when beta==0
 *          - outQueueCR_, outQueueCI_ (VECOUT): C tile real/imag, V→MTE3 sync via TQue
 *          - calcBufR_, calcBufI_ (VECCALC): Scratch for complex multiply intermediates
 *
 *        Conjugate handling:
 *          - isConjA: negate A imaginary after DeQue (PipeBarrier<PIPE_V> before compute)
 *          - isConjB: negate B imaginary after load (PipeBarrier<PIPE_V> before compute)
 *
 *        Single path: one column per iteration (complex interleaved prevents multi-col batching)
 *        Multi-core decomposition: 2D grid (colBlocks x mBlocks)
 */

#include "kernel_operator.h"
#include "cgeam_kernel.h"
#include "cgeam_tiling_data.h"

namespace {

using namespace AscendC;

// ============================================================================
// CgeamAIV: operator class for Cgeam complex vector kernel
// ============================================================================
class CgeamAIV {
public:
    __aicore__ inline CgeamAIV() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTileLoop();
    __aicore__ inline void ComputeCoreRanges(const CgeamTilingData& tiling);
    __aicore__ inline void CopyInCplxA(
        uint32_t col, uint32_t rowStart, uint32_t curM, int64_t ntCplxStride, int64_t tCplxStrideA);
    __aicore__ inline void CopyInCplxB(
        uint32_t col, uint32_t rowStart, uint32_t curM, int64_t ntCplxStride, int64_t tCplxStrideB);
    __aicore__ inline void ComputeCplx(uint32_t curM);
    __aicore__ inline void CopyOutCplx(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t storeStride);

    // GM addresses
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;

    // A input queues (separate real/imag for clean TQue sync)
    TQue<TPosition::VECIN, 2> inQueueAR_; // A real tile (EnQue signals MTE2 done)
    TQue<TPosition::VECIN, 2> inQueueAI_; // A imag tile (EnQue signals MTE2 done)

    // B input: loaded via TBuf (same pattern as sgeam)
    TBuf<TPosition::VECCALC> calcBufBR_; // B real tile (also serves as scratch when beta==0)
    TBuf<TPosition::VECCALC> calcBufBI_; // B imag tile (also serves as scratch when beta==0)

    // C output queues
    TQue<TPosition::VECOUT, 2> outQueueCR_; // C real tile (EnQue signals Vector done)
    TQue<TPosition::VECOUT, 2> outQueueCI_; // C imag tile (EnQue signals Vector done)

    // Scratch for complex multiply
    TBuf<TPosition::VECCALC> calcBufR_; // Temp for alpha_i*A or beta_i*B real component
    TBuf<TPosition::VECCALC> calcBufI_; // Temp for alpha_i*A or beta_i*B imag component

    // Tiling fields cached from CgeamTilingData
    uint32_t m_;
    uint32_t n_;

    uint32_t alphaIsZero_;
    float alphaR_;
    float alphaI_;
    aclblasOperation_t opA_;
    uint32_t lda_;
    bool isConjA_;

    uint32_t betaIsZero_;
    float betaR_;
    float betaI_;
    aclblasOperation_t opB_;
    uint32_t ldb_;
    bool isConjB_;

    uint32_t ldc_;
    uint32_t tileM_;

    // Per-core tile ranges
    uint32_t colStart_;
    uint32_t colEnd_;
    uint32_t mTileStart_;
    uint32_t mTileEnd_;
};

// ============================================================================
// ComputeCoreRanges: decode 2D block index into per-core column and m-tile ranges.
// On invalid range, sets colEnd_ or mTileEnd_ to 0 to signal early exit in Process().
// ============================================================================
__aicore__ inline void CgeamAIV::ComputeCoreRanges(const CgeamTilingData& tiling)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t mBlocks = tiling.mBlocks;
    uint32_t colBlock = blockIdx / mBlocks;
    uint32_t mBlock = blockIdx % mBlocks;

    // Calculate column range
    if (colBlock < tiling.remainder) {
        colStart_ = colBlock * (tiling.perCoreN + 1);
        colEnd_ = colStart_ + tiling.perCoreN + 1;
    } else {
        colStart_ = colBlock * tiling.perCoreN + tiling.remainder;
        colEnd_ = colStart_ + tiling.perCoreN;
    }

    // Calculate m-tile range
    if (mBlock < tiling.mTileRemainder) {
        mTileStart_ = mBlock * (tiling.perCoreMTile + 1);
        mTileEnd_ = mTileStart_ + tiling.perCoreMTile + 1;
    } else {
        mTileStart_ = mBlock * tiling.perCoreMTile + tiling.mTileRemainder;
        mTileEnd_ = mTileStart_ + tiling.perCoreMTile;
    }

    // Validate column range
    if (colStart_ >= n_ || colStart_ >= colEnd_) {
        colEnd_ = 0; // Signal Process() to return early
        return;
    }
    if (colEnd_ > n_) {
        colEnd_ = n_;
    }

    // Validate m-tile range
    uint32_t totalMTiles = (m_ + tileM_ - 1) / tileM_;
    if (mTileStart_ >= totalMTiles || mTileStart_ >= mTileEnd_) {
        mTileEnd_ = 0; // Signal Process() to return early
        return;
    }
    if (mTileEnd_ > totalMTiles) {
        mTileEnd_ = totalMTiles;
    }
}

// ============================================================================
// Init: setup global tensors, compute per-core ranges, allocate UB buffers.
// TPipe pointer passed from kernel entry (R3: TPipe must not be a member).
// ============================================================================
__aicore__ inline void CgeamAIV::Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe)
{
    // Cache tiling fields to member variables
    m_ = tiling.m;
    n_ = tiling.n;
    lda_ = tiling.lda;
    ldb_ = tiling.ldb;
    ldc_ = tiling.ldc;
    tileM_ = tiling.tileM;
    opA_ = tiling.opA;
    opB_ = tiling.opB;
    alphaIsZero_ = tiling.alphaIsZero;
    betaIsZero_ = tiling.betaIsZero;
    alphaR_ = tiling.alphaR;
    alphaI_ = tiling.alphaI;
    betaR_ = tiling.betaR;
    betaI_ = tiling.betaI;
    isConjA_ = (opA_ == ACLBLAS_OP_C);
    isConjB_ = (opB_ == ACLBLAS_OP_C);

    // Set GM buffers
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));

    // Compute per-core column and m-tile ranges
    ComputeCoreRanges(tiling);
    if (colEnd_ == 0 || mTileEnd_ == 0) {
        return;
    }

    // Buffer size: tileM floats per buffer
    uint32_t bufSize = tileM_ * sizeof(float);

    // A input queues (double-buffer, num=2 for MTE2<->V pipeline overlap)
    pipe->InitBuffer(inQueueAR_, 2, bufSize);
    pipe->InitBuffer(inQueueAI_, 2, bufSize);

    // C output queues (double-buffer, num=2 for V<->MTE3 pipeline overlap)
    pipe->InitBuffer(outQueueCR_, 2, bufSize);
    pipe->InitBuffer(outQueueCI_, 2, bufSize);

    // B input (TBuf, allocated when beta != 0)
    if (!betaIsZero_) {
        pipe->InitBuffer(calcBufBR_, bufSize);
        pipe->InitBuffer(calcBufBI_, bufSize);
    }

    // Complex multiply scratch (only needed when alpha != 0)
    if (!alphaIsZero_) {
        pipe->InitBuffer(calcBufR_, bufSize);
        pipe->InitBuffer(calcBufI_, bufSize);
    }
}

// ============================================================================
// Process: validate ranges, dispatch to tile loop
// ============================================================================
__aicore__ inline void CgeamAIV::Process()
{
    if (colStart_ >= colEnd_ || mTileStart_ >= mTileEnd_) {
        return;
    }
    ProcessTileLoop();
}

// ============================================================================
// CopyInCplxA: load A real and imag tiles through their respective VECIN queues.
// ============================================================================
__aicore__ inline void CgeamAIV::CopyInCplxA(
    uint32_t col, uint32_t rowStart, uint32_t curM, int64_t ntCplxStride, int64_t tCplxStrideA)
{
    // CopyIn A real (MTE2)
    LocalTensor<float> aR = inQueueAR_.AllocTensor<float>();
    if (!alphaIsZero_) {
        uint64_t offsetBase;
        int64_t stride;
        if (opA_ == ACLBLAS_OP_N) {
            offsetBase = static_cast<uint64_t>(col) * static_cast<uint64_t>(lda_) + rowStart;
            stride = ntCplxStride;
        } else {
            offsetBase = static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(lda_);
            stride = tCplxStrideA;
        }
        DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), stride, 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(aR, aGm_[2 * offsetBase], cp, np);
    }
    inQueueAR_.EnQue(aR); // Signal MTE2 done for A real

    // CopyIn A imag (MTE2)
    LocalTensor<float> aI = inQueueAI_.AllocTensor<float>();
    if (!alphaIsZero_) {
        uint64_t offsetBase;
        int64_t stride;
        if (opA_ == ACLBLAS_OP_N) {
            offsetBase = static_cast<uint64_t>(col) * static_cast<uint64_t>(lda_) + rowStart;
            stride = ntCplxStride;
        } else {
            offsetBase = static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(lda_);
            stride = tCplxStrideA;
        }
        DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), stride, 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(aI, aGm_[2 * offsetBase + 1], cp, np);
    }
    inQueueAI_.EnQue(aI); // Signal MTE2 done for A imag
}

// ============================================================================
// CopyInCplxB: load B real and imag tiles, and apply conjugate if needed.
// ============================================================================
__aicore__ inline void CgeamAIV::CopyInCplxB(
    uint32_t col, uint32_t rowStart, uint32_t curM, int64_t ntCplxStride, int64_t tCplxStrideB)
{
    if (!betaIsZero_) {
        LocalTensor<float> bR = calcBufBR_.Get<float>();
        LocalTensor<float> bI = calcBufBI_.Get<float>();

        uint64_t offsetBase;
        int64_t stride;
        if (opB_ == ACLBLAS_OP_N) {
            offsetBase = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldb_) + rowStart;
            stride = ntCplxStride;
        } else {
            offsetBase = static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(ldb_);
            stride = tCplxStrideB;
        }

        DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), stride, 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(bR, bGm_[2 * offsetBase], cp, np);
        DataCopyPad<float, PaddingMode::Compact>(bI, bGm_[2 * offsetBase + 1], cp, np);
        event_t eMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eMte2V_);
        WaitFlag<HardEvent::MTE2_V>(eMte2V_);

        if (isConjB_) {
            Muls<float>(bI, bI, -1.0f, curM);
            PipeBarrier<PIPE_V>();
        }
    }
}

// ============================================================================
// ComputeCplx: Phase 1 (C = alpha * A) + Phase 2 (C += beta * B) complex multiply.
// ============================================================================
__aicore__ inline void CgeamAIV::ComputeCplx(uint32_t curM)
{
    // DeQue A waits for MTE2 -> V sync
    LocalTensor<float> aRv = inQueueAR_.DeQue<float>();
    LocalTensor<float> aIv = inQueueAI_.DeQue<float>();

    // Conjugate A: negate imaginary part after DeQue
    if (isConjA_) {
        Muls<float>(aIv, aIv, -1.0f, curM);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> cR = outQueueCR_.AllocTensor<float>();
    LocalTensor<float> cI = outQueueCI_.AllocTensor<float>();

    if (alphaIsZero_) {
        // Zero-initialize C (alpha == 0: A not referenced)
        Duplicate<float>(cR, 0.0f, curM);
        Duplicate<float>(cI, 0.0f, curM);
    } else {
        // Scratch buffers for complex multiply
        LocalTensor<float> tmpR = calcBufR_.Get<float>();
        LocalTensor<float> tmpI = calcBufI_.Get<float>();

        // Phase 1: C = alpha * A (complex multiplication)
        // C_r = alphaR * A_r - alphaI * A_i
        // C_i = alphaR * A_i + alphaI * A_r
        Muls<float>(tmpR, aRv, alphaI_, curM); // tmpR = alphaI * A_r
        Muls<float>(tmpI, aIv, alphaI_, curM); // tmpI = alphaI * A_i
        Muls<float>(cR, aRv, alphaR_, curM);   // cR = alphaR * A_r
        Muls<float>(cI, aIv, alphaR_, curM);   // cI = alphaR * A_i
        Sub<float>(cR, cR, tmpI, curM);        // cR -= alphaI * A_i
        Add<float>(cI, cI, tmpR, curM);        // cI += alphaI * A_r
    }

    if (!betaIsZero_) {
        // Phase 2: C += beta * B (complex multiplication)
        LocalTensor<float> bR = calcBufBR_.Get<float>();
        LocalTensor<float> bI = calcBufBI_.Get<float>();
        // When alpha != 0, use calcBufR_/I_ as scratch; when alpha == 0,
        // reuse A input buffers (aRv/aIv) as scratch since Phase 1 was skipped.
        LocalTensor<float> tmpR = alphaIsZero_ ? aRv : calcBufR_.Get<float>();
        LocalTensor<float> tmpI = alphaIsZero_ ? aIv : calcBufI_.Get<float>();
        // C_r += betaR * B_r - betaI * B_i
        // C_i += betaR * B_i + betaI * B_r
        Muls<float>(tmpR, bR, betaR_, curM); // tmpR = betaR * B_r
        Add<float>(cR, cR, tmpR, curM);      // cR += betaR * B_r
        Muls<float>(tmpI, bI, betaR_, curM); // tmpI = betaR * B_i
        Add<float>(cI, cI, tmpI, curM);      // cI += betaR * B_i
        Muls<float>(tmpR, bR, betaI_, curM); // tmpR = betaI * B_r
        Add<float>(cI, cI, tmpR, curM);      // cI += betaI * B_r
        Muls<float>(tmpI, bI, betaI_, curM); // tmpI = betaI * B_i
        Sub<float>(cR, cR, tmpI, curM);      // cR -= betaI * B_i
    }

    outQueueCR_.EnQue(cR); // Signal Vector done for C real
    outQueueCI_.EnQue(cI); // Signal Vector done for C imag
    inQueueAR_.FreeTensor(aRv);
    inQueueAI_.FreeTensor(aIv);
}

// ============================================================================
// CopyOutCplx: store C real and imag tiles to global memory (interleaved).
// ============================================================================
__aicore__ inline void CgeamAIV::CopyOutCplx(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t storeStride)
{
    // CopyOut C real (MTE3): DeQue waits for V -> MTE3 sync
    LocalTensor<float> cRo = outQueueCR_.DeQue<float>();
    {
        const uint64_t offsetBase = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldc_) + rowStart;
        DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), 0, storeStride, 0};
        DataCopyPad<float, PaddingMode::Compact>(cGm_[2 * offsetBase], cRo, cp);
    }
    outQueueCR_.FreeTensor(cRo);

    // CopyOut C imag (MTE3)
    LocalTensor<float> cIo = outQueueCI_.DeQue<float>();
    {
        const uint64_t offsetBase = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldc_) + rowStart;
        DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), 0, storeStride, 0};
        DataCopyPad<float, PaddingMode::Compact>(cGm_[2 * offsetBase + 1], cIo, cp);
    }
    outQueueCI_.FreeTensor(cIo);
}

// ============================================================================
// ProcessTileLoop: complex geam tile processing
// Each tile: CopyIn A -> EnQue -> CopyIn B -> Compute -> EnQue C -> CopyOut C
// ============================================================================
__aicore__ inline void CgeamAIV::ProcessTileLoop()
{
    // Pre-compute strides for complex strided access
    // NoTrans complex: adjacent r (or i) elements are 2 floats apart → srcStride = sizeof(float)
    const int64_t ntCplxStride = static_cast<int64_t>(sizeof(float));
    // Trans/ConjTrans complex: each row = ld complex = 2*ld floats → gap = (2*ld-1)*sizeof(float)
    const int64_t tCplxStrideA = static_cast<int64_t>(2 * lda_ - 1) * static_cast<int64_t>(sizeof(float));
    const int64_t tCplxStrideB = static_cast<int64_t>(2 * ldb_ - 1) * static_cast<int64_t>(sizeof(float));
    // Store: interleaved complex write, dstStride gap = sizeof(float)
    const int64_t storeStride = static_cast<int64_t>(sizeof(float));

    for (uint32_t col = colStart_; col < colEnd_; ++col) {
        for (uint32_t mTile = mTileStart_; mTile < mTileEnd_; ++mTile) {
            const uint32_t rowStart = mTile * tileM_;
            uint32_t rowEnd = rowStart + tileM_;
            if (rowEnd > m_) {
                rowEnd = m_;
            }
            const uint32_t curM = rowEnd - rowStart;

            CopyInCplxA(col, rowStart, curM, ntCplxStride, tCplxStrideA);
            CopyInCplxB(col, rowStart, curM, ntCplxStride, tCplxStrideB);
            ComputeCplx(curM);
            CopyOutCplx(col, rowStart, curM, storeStride);
        }
    }
}

} // namespace

// ============================================================================
// Kernel entry point
// ============================================================================
extern "C" __global__ __aicore__ void cgeam_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.m == 0 || tiling.n == 0) {
        return;
    }

    TPipe pipe;
    CgeamAIV op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

void cgeam_kernel_do(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, uint32_t numBlocks, aclrtStream stream)
{
    cgeam_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
}
