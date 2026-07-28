/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyrk_kernel.cpp
 * \brief SSYRK Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: GEMM kernel   (AIC-only, tensor_api) - computes temp = op(A) * op(A)^T
 *                Uses DNExtLayoutPtn for on-the-fly transpose during GM→L1 (no separate transpose kernel).
 *        Phase 2: Scale kernel  (AIV-only, SIMD) - C = alpha * temp + beta * C
 *                Exploits temp symmetry: reads temp with swapped (iBase,jBase) to get column-major
 *                layout without Gather transpose.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "ssyrk_tiling_data.h"
#include "ssyrk_kernel.h"

namespace te = AscendC::Te;

constexpr int64_t L0A_SIZE = 64 * 1024;          // [general] DAV_3510 L0A/L0B capacity per core
constexpr int64_t L0C_SIZE = 256 * 1024;          // [general] DAV_3510 L0C capacity per core
constexpr int64_t L1_SIZE = static_cast<int64_t>(SYRK_ARCH35_L1_SIZE_BYTES);

constexpr uint64_t SYRK_FP32_C0 = 8;
constexpr uint64_t SYRK_FRACTAL = 16;
constexpr uint64_t SYRK_L0C_C0 = 16;
constexpr uint64_t SYRK_L0_BUF_MASK = 0x1;
constexpr uint64_t SYRK_HALF_L0_SIZE = L0A_SIZE / 2;
constexpr uint64_t SYRK_L0C_BUF_MASK = 0x1;
constexpr uint64_t SYRK_HALF_L0C_SIZE = L0C_SIZE / 2;

using namespace AscendC;

// ================================================================================
// Phase 1: GEMM Kernel (AIC-only, tensor_api)
// Computes temp = op(A) * op(A)^T using a single gmA for both operands.
// A is column-major: A[i][j] at address j*lda + i.
//   NDExtLayoutPtn(R,C): GM treated as row-major, element [r][c] at r*C + c.
//   DNExtLayoutPtn(R,C): GM treated as column-major, element [r][c] at c*R + r.
//
// trans=N: temp = A(N×K) × A^T(K×N)
//   left  = A   → DNExt(lda, K): [n][k] at k*lda+n = A[n][k] ✓ (slice lda→N)
//   right = A^T → NDExt(K, lda): [k][n] at k*lda+n = A[n][k] = A^T[k][n] ✓ (slice lda→N)
//
// trans=T: temp = A^T(N×K) × A(K×N)
//   left  = A^T → NDExt(N, lda): [n][k] at n*lda+k = A[k][n] = A^T[n][k] ✓ (slice lda→K)
//   right = A   → DNExt(lda, N): [k][n] at n*lda+k = A[k][n] ✓ (slice lda→K)
// ================================================================================

template <typename CopyAtom, typename TensorGM, typename TensorL1>
__aicore__ inline void SyrkCopyGM2L1(
    CopyAtom copyGM2L1, TensorGM gmTensor, TensorL1 tensorL1,
    uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void SyrkL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0,
    uint64_t iter0, uint64_t baseK,
    uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);

    uint64_t l0cOffset = l0cBufId * SYRK_HALF_L0C_SIZE;
    auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_L0C_C0>>(curML1, nL0);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & SYRK_L0_BUF_MASK;
        uint64_t l0Offset = SYRK_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.Slice(te::MakeCoord(kL0Offset, 0), te::MakeShape(curKL0, nL0));
        te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isFirstK = (iter0 == 0 && iter1 == 0);
        te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0),
            static_cast<uint16_t>(curKL0), 0, isFirstK};
        te::Mmad(mmadAtom.with(mmadParams),
            tensorL0C, tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

template <typename CopyAtom, typename TensorA, typename TensorB, typename TensorAL1, typename TensorBL1>
__aicore__ inline void SyrkGemmProcessKChunk(
    CopyAtom copyGM2L1, TensorA gmLeftTensor, TensorB gmRightTensor,
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t l1BufId,
    uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK,
    uint64_t mL0, uint64_t nL0,
    uint64_t iter0,
    uint64_t baseK,
    uint64_t& l0PingPong, uint64_t l0cBufId)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    SyrkCopyGM2L1(copyGM2L1, gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    SyrkCopyGM2L1(copyGM2L1, gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    SyrkL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0,
        iter0, baseK, l0PingPong, l0cBufId);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorA, typename TensorB>
__aicore__ inline void SyrkProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK,
    uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    for (uint32_t kOff = 0; kOff < K; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & (SYRK_ARCH35_L1_BUF_NUM - 1);
        uint64_t l1OffsetA = l1BufId * (L1_SIZE / SYRK_ARCH35_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, SYRK_FRACTAL) * RoundUp<uint64_t>(curK, SYRK_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_FP32_C0>>(curK, nL0));
        SyrkGemmProcessKChunk(copyGM2L1, gmLeftTensor, gmRightTensor,
            tensorAL1, tensorBL1, l1BufId,
            mOff, nOff, kOff, curK, mL0, nL0,
            kOff / tileKChunk, baseK, l0PingPong, l0cBufId);
        abL1LoopCnt++;
    }
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkProcessTile(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd,
    uint32_t tileM, uint32_t tileN,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t& l0cPingPong)
{
    auto copyL0C2GMAtom = te::MakeCopy(te::CopyL0C2GM{});

    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            uint64_t l0cBufId = l0cPingPong & SYRK_L0C_BUF_MASK;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            SyrkProcessKChunks(gmLeftTensor, gmRightTensor,
                K, tileKChunk, baseK,
                mOff, nOff, curTileM, curTileN, abL1LoopCnt, l0PingPong, l0cBufId);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            uint64_t l0cOffset = l0cBufId * SYRK_HALF_L0C_SIZE;
            auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_L0C_C0>>(curTileM, curTileN);
            auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
            auto gmBlockC = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curTileM, curTileN));
            copyL0C2GMAtom.Call(gmBlockC, tensorL0C, te::FixpipeParams{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            l0cPingPong++;
        }
    }
}

// trans=N: left=DNExt(lda,K) right=NDExt(K,lda)
template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkGemmKernelImpl(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    const SyrkGemmTilingData& tiling)
{
    uint32_t n = tiling.n;
    uint32_t K = tiling.k;

    uint32_t divM = CeilDiv<uint32_t>(n, tiling.singleCoreM);
    uint32_t divN = CeilDiv<uint32_t>(n, tiling.singleCoreN);
    uint64_t totalTiles = static_cast<uint64_t>(divM) * static_cast<uint64_t>(divN);

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

    uint64_t l0PingPong = 0;
    uint64_t l0cPingPong = 0;
    uint64_t abL1LoopCnt = 0;

    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += AscendC::GetBlockNum()) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }
        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        SyrkProcessTile(gmLeftTensor, gmRightTensor, gmTempTensor,
            K, tiling.tileKChunk, SYRK_ARCH35_BASE_K,
            mStart, Min<uint32_t>(mStart + tiling.singleCoreM, n),
            nStart, Min<uint32_t>(nStart + tiling.singleCoreN, n),
            tiling.tileM, tiling.tileN, abL1LoopCnt, l0PingPong, l0cPingPong);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
}

extern "C" __global__ __aicore__ void syrk_gemm_kernel(
    GM_ADDR gmA, GM_ADDR gmTemp,
    const SyrkGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));

    if (tiling.isTransN != 0) {
        // trans=N: left=DNExt(lda,K) reads A[n][k] from column-major; right=NDExt(K,lda) reads A^T
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.lda, tiling.k));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.k, tiling.lda));
        SyrkGemmKernelImpl(gmLeftTensor, gmRightTensor, gmTempTensor, tiling);
    } else {
        // trans=T: left=NDExt(N,lda) reads A^T; right=DNExt(lda,N) reads A[k][n]
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.lda));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.lda, tiling.n));
        SyrkGemmKernelImpl(gmLeftTensor, gmRightTensor, gmTempTensor, tiling);
    }
}

void syrk_gemm_kernel_do(
    GM_ADDR gmA, GM_ADDR gmTemp,
    const SyrkGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    syrk_gemm_kernel<<<numBlocks, nullptr, stream>>>(
        gmA, gmTemp, tiling);
}

// ================================================================================
// Phase 2: Scale Kernel (AIV-only, SIMD)
// C = alpha * temp + beta * C (column-major C, row-major temp)
// Only updates the uplo triangle (UPPER: j>=i, LOWER: j<=i).
// k=0 also only updates uplo triangle (no temp read, C=beta*C).
// Processes 64×64 blocks; skips blocks entirely outside uplo triangle.
//
// Transpose elimination via temp symmetry:
//   temp = A * A^T is symmetric, so temp[i][j] = temp[j][i].
//   Instead of reading temp[iBase..iBase+rows][jBase..jBase+cols] (row-major → needs
//   Gather transpose), we read temp[jBase..jBase+cols][iBase..iBase+rows] with swapped
//   coordinates. By symmetry, temp[jBase+c][iBase+r] = temp[iBase+r][jBase+c], which is
//   exactly the value we need. The DataCopyPad reads `cols` rows of `rows` elements each,
//   producing a column-major UB layout that matches cInUb — no Gather transpose needed.
// ================================================================================

constexpr uint32_t SYRK_SCALE_BLOCK = SYRK_ARCH35_SCALE_BLOCK;
constexpr uint32_t SYRK_SCALE_UB_FLOATS = SYRK_SCALE_BLOCK * SYRK_SCALE_BLOCK;

class SyrkScaleAIV {
public:
    __aicore__ inline explicit SyrkScaleAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR gmTemp, GM_ADDR gmC, const SyrkScaleTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    SyrkScaleTilingData tiling_;
    GlobalTensor<float> tempGM_;
    GlobalTensor<float> cGM_;
    TBuf<TPosition::VECIN> tempBuf_;
    TBuf<TPosition::VECIN> cInBuf_;
    TBuf<TPosition::VECOUT> cOutBuf_;
    uint32_t rowStart_;
    uint32_t rowEnd_;

    __aicore__ inline void ProcessBlock(uint32_t iBase, uint32_t jBase,
        uint32_t rows, uint32_t cols);
    __aicore__ inline void ApplyScale(LocalTensor<float>& cUb, LocalTensor<float>& cInUb,
        LocalTensor<float>& tempUb, uint32_t offset, int32_t count,
        bool skipTemp, bool isBetaZero, float alpha, float beta);
    __aicore__ inline void ComputeScaleResult(uint32_t iBase, uint32_t jBase,
        uint32_t rows, uint32_t cols, uint32_t ubColStride,
        LocalTensor<float>& cUb, LocalTensor<float>& cInUb,
        LocalTensor<float>& tempUb, bool skipTemp, bool isBetaZero,
        float alpha, float beta);
};

__aicore__ inline void SyrkScaleAIV::Init(
    GM_ADDR gmTemp, GM_ADDR gmC, const SyrkScaleTilingData& tiling)
{
    tiling_ = tiling;
    tempGM_.SetGlobalBuffer((__gm__ float*)gmTemp);
    cGM_.SetGlobalBuffer((__gm__ float*)gmC);

    pipe_.InitBuffer(tempBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cInBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cOutBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));

    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_.rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_.rowsPerCore, tiling_.n);
}

__aicore__ inline void SyrkScaleAIV::ApplyScale(
    LocalTensor<float>& cUb, LocalTensor<float>& cInUb, LocalTensor<float>& tempUb,
    uint32_t offset, int32_t count, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    if (!skipTemp) {
        if (isBetaZero) {
            Muls(cUb[offset], tempUb[offset], alpha, count);
        } else {
            Muls(cUb[offset], cInUb[offset], beta, count);
            Axpy(cUb[offset], tempUb[offset], alpha, count);
        }
    } else {
        Muls(cUb[offset], cInUb[offset], beta, count);
    }
}

__aicore__ inline void SyrkScaleAIV::ComputeScaleResult(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols,
    uint32_t ubColStride, LocalTensor<float>& cUb, LocalTensor<float>& cInUb,
    LocalTensor<float>& tempUb, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    bool uploUpper = (tiling_.uploMode == ACLBLAS_UPPER);

    bool isFullInterior = uploUpper
        ? (jBase >= iBase + rows)
        : (jBase + cols <= iBase);

    if (isFullInterior) {
        int32_t totalCount = static_cast<int32_t>(ubColStride * cols);
        ApplyScale(cUb, cInUb, tempUb, 0, totalCount, skipTemp, isBetaZero, alpha, beta);
        return;
    }

    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        uint32_t colOffset = c * ubColStride;

        uint32_t uploCount;
        if (uploUpper) {
            uploCount = (absJ >= iBase) ? Min(absJ - iBase + 1, rows) : 0;
        } else {
            uploCount = (absJ < iBase) ? rows : (rows - (absJ - iBase));
        }
        uint32_t nonUploCount = rows - uploCount;

        if (nonUploCount > 0 && uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(rows));
        }

        uint32_t computeCount = uploUpper ? uploCount : rows;
        if (computeCount > 0) {
            ApplyScale(cUb, cInUb, tempUb, colOffset,
                static_cast<int32_t>(computeCount), skipTemp, isBetaZero, alpha, beta);
        }

        if (nonUploCount > 0 && !uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(nonUploCount));
        }
    }
}

__aicore__ inline void SyrkScaleAIV::ProcessBlock(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = tiling_.alphaVal;
    float beta = tiling_.betaVal;
    bool skipTemp = (tiling_.isAlphaZero || tiling_.isKZero);
    bool isBetaZero = tiling_.isBetaZero;

    uint32_t ldc = tiling_.ldc;
    uint32_t tempRowStride = tiling_.tempRowStride;
    uint32_t ubColStride = RoundUp<uint32_t>(rows, SYRK_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> cUb = cOutBuf_.Get<float>();
    LocalTensor<float> cInUb = cInBuf_.Get<float>();

    // Read C columns [jBase, jBase+cols), rows [iBase, iBase+rows) — column-major, contiguous
    int64_t cSrcStride = static_cast<int64_t>(ldc - rows) * sizeof(float);
    DataCopyExtParams cpC{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), cSrcStride, 0, 0};
    DataCopyPadExtParams<float> ppC{true, 0, 0, 0.0f};
    int64_t cGmOffset = static_cast<int64_t>(jBase) * ldc + iBase;
    DataCopyPad(cInUb, cGM_[cGmOffset], cpC, ppC);

    LocalTensor<float> tempUb = tempBuf_.Get<float>();

    if (!skipTemp) {
        // Read temp with swapped coordinates: rows [jBase, jBase+cols), cols [iBase, iBase+rows).
        // By temp symmetry: temp[jBase+c][iBase+r] = temp[iBase+r][jBase+c] — correct value,
        // and the DataCopyPad produces column-major UB layout matching cInUb. No transpose needed.
        int64_t tempSrcStride = static_cast<int64_t>(tempRowStride - rows) * sizeof(float);
        DataCopyExtParams cpTemp{static_cast<uint16_t>(cols),
            static_cast<uint32_t>(rows * sizeof(float)),
            tempSrcStride, 0, 0};
        DataCopyPadExtParams<float> ppTemp{true, 0, 0, 0.0f};
        int64_t tempGmOffset = static_cast<int64_t>(jBase) * tempRowStride + iBase;
        DataCopyPad(tempUb, tempGM_[tempGmOffset], cpTemp, ppTemp);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

    ComputeScaleResult(iBase, jBase, rows, cols, ubColStride, cUb, cInUb,
        tempUb, skipTemp, isBetaZero, alpha, beta);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    DataCopyExtParams cpOut{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), 0,
        cSrcStride, 0};
    DataCopyPad(cGM_[cGmOffset], cUb, cpOut);
}

__aicore__ inline void SyrkScaleAIV::Process()
{
    if (rowStart_ >= rowEnd_) {
        return;
    }
    uint32_t n = tiling_.n;
    bool uploUpper = (tiling_.uploMode == ACLBLAS_UPPER);

    for (uint32_t iBase = rowStart_; iBase < rowEnd_; iBase += SYRK_SCALE_BLOCK) {
        uint32_t rows = Min<uint32_t>(SYRK_SCALE_BLOCK, rowEnd_ - iBase);
        uint32_t iEnd = iBase + rows - 1;

        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? n : Min(iEnd + 1, n);

        for (uint32_t jBase = jStart; jBase < jLimit; jBase += SYRK_SCALE_BLOCK) {
            uint32_t cols = Min<uint32_t>(SYRK_SCALE_BLOCK, jLimit - jBase);
            ProcessBlock(iBase, jBase, rows, cols);
        }
    }
}

extern "C" __global__ __aicore__ void syrk_scale_kernel(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SyrkScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SyrkScaleAIV op(pipe);
    op.Init(gmTemp, gmC, tiling);
    op.Process();
}

void syrk_scale_kernel_do(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SyrkScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    syrk_scale_kernel<<<numBlocks, nullptr, stream>>>(
        gmTemp, gmC, tiling);
}
