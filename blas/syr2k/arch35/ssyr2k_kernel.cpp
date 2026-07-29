/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyr2k_kernel.cpp
 * \brief SSYR2K Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: GEMM kernel   (AIC-only, tensor_api) - computes temp1 = op(A) * op(B)^T
 *                                                         and temp2 = op(B) * op(A)^T
 *                Uses DNExtLayoutPtn for on-the-fly transpose during GM->L1 (no separate transpose kernel).
 *        Phase 2: Scale kernel  (AIV-only, SIMD) - C = alpha * (temp1 + temp2) + beta * C
 *                Exploits temp1+temp2 symmetry: reads both with swapped (iBase,jBase) to get
 *                column-major layout without Gather transpose.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "ssyr2k_tiling_data.h"
#include "ssyr2k_kernel.h"

namespace te = AscendC::Te;

constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L0C_SIZE = 256 * 1024;
constexpr int64_t L1_SIZE = static_cast<int64_t>(SSYR2K_ARCH35_L1_SIZE_BYTES);

constexpr uint64_t SSYR2K_FP32_C0 = 8;
constexpr uint64_t SSYR2K_FRACTAL = 16;
constexpr uint64_t SSYR2K_L0C_C0 = 16;
constexpr uint64_t SSYR2K_L0_BUF_MASK = 0x1;
constexpr uint64_t SSYR2K_HALF_L0_SIZE = L0A_SIZE / 2;
constexpr uint64_t SSYR2K_L0C_BUF_MASK = 0x1;
constexpr uint64_t SSYR2K_HALF_L0C_SIZE = L0C_SIZE / 2;

using namespace AscendC;

// ================================================================================
// Phase 1: GEMM Kernel (AIC-only, tensor_api)
// Computes temp = left x right using DNExt/NDExt for on-the-fly transpose.
//
// A/B are column-major: A[i][j] at address j*lda + i, B[i][j] at address j*ldb + i.
//   NDExtLayoutPtn(R,C): GM treated as row-major, element [r][c] at r*C + c.
//   DNExtLayoutPtn(R,C): GM treated as column-major, element [r][c] at c*R + r.
//
// GEMM1: temp1 = op(A) * op(B)^T   (left=A, right=B)
// GEMM2: temp2 = op(B) * op(A)^T   (left=B, right=A)
//
// trans=N: op(X) = X (n*k), op(X)^T = X^T (k*n)
//   left  = X    -> DNExt(leftLd, K):  [n][k] at k*leftLd+n = X[n][k]
//   right = Y^T  -> NDExt(K, rightLd): [k][n] at k*rightLd+n = Y[n][k] = Y^T[k][n]
//
// trans=T: op(X) = X^T (n*k), op(X)^T = X (k*n)
//   left  = X^T  -> NDExt(N, leftLd):  [n][k] at n*leftLd+k = X[k][n] = X^T[n][k]
//   right = Y    -> DNExt(rightLd, N): [k][n] at n*rightLd+k = Y[k][n]
// ================================================================================

template <typename CopyAtom, typename TensorGM, typename TensorL1>
__aicore__ inline void Ssyr2kCopyGM2L1(
    CopyAtom copyGM2L1, TensorGM gmTensor, TensorL1 tensorL1,
    uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void Ssyr2kL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0,
    uint64_t iter0, uint64_t baseK,
    uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);

    uint64_t l0cOffset = l0cBufId * SSYR2K_HALF_L0C_SIZE;
    auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SSYR2K_L0C_C0>>(curML1, nL0);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & SSYR2K_L0_BUF_MASK;
        uint64_t l0Offset = SSYR2K_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SSYR2K_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SSYR2K_FP32_C0>>(curKL0, nL0);
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
__aicore__ inline void Ssyr2kGemmProcessKChunk(
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
    Ssyr2kCopyGM2L1(copyGM2L1, gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    Ssyr2kCopyGM2L1(copyGM2L1, gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    Ssyr2kL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0,
        iter0, baseK, l0PingPong, l0cBufId);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorA, typename TensorB>
__aicore__ inline void Ssyr2kProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK,
    uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    for (uint32_t kOff = 0; kOff < K; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & (SSYR2K_ARCH35_L1_BUF_NUM - 1);
        uint64_t l1OffsetA = l1BufId * (L1_SIZE / SSYR2K_ARCH35_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, SSYR2K_FRACTAL)
            * RoundUp<uint64_t>(curK, SSYR2K_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn,
                AscendC::Std::Int<SSYR2K_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn,
                AscendC::Std::Int<SSYR2K_FP32_C0>>(curK, nL0));
        Ssyr2kGemmProcessKChunk(copyGM2L1, gmLeftTensor, gmRightTensor,
            tensorAL1, tensorBL1, l1BufId,
            mOff, nOff, kOff, curK, mL0, nL0,
            kOff / tileKChunk, baseK, l0PingPong, l0cBufId);
        abL1LoopCnt++;
    }
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void Ssyr2kProcessTile(
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
            uint64_t l0cBufId = l0cPingPong & SSYR2K_L0C_BUF_MASK;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            Ssyr2kProcessKChunks(gmLeftTensor, gmRightTensor,
                K, tileKChunk, baseK,
                mOff, nOff, curTileM, curTileN, abL1LoopCnt, l0PingPong, l0cBufId);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            uint64_t l0cOffset = l0cBufId * SSYR2K_HALF_L0C_SIZE;
            auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SSYR2K_L0C_C0>>(curTileM, curTileN);
            auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
            auto gmBlockC = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curTileM, curTileN));
            copyL0C2GMAtom.Call(gmBlockC, tensorL0C, te::FixpipeParams{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            l0cPingPong++;
        }
    }
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void Ssyr2kGemmKernelImpl(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    const Ssyr2kGemmTilingData& tiling)
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
        // Zigzag (snake) traversal: reverse N direction on odd M rows to improve L2 cache locality
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }
        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        Ssyr2kProcessTile(gmLeftTensor, gmRightTensor, gmTempTensor,
            K, tiling.tileKChunk, SSYR2K_ARCH35_BASE_K,
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

extern "C" __global__ __aicore__ void ssyr2k_gemm_kernel(
    GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmTemp,
    const Ssyr2kGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));

    if (tiling.isTransN != 0) {
        // trans=N: left=DNExt(leftLd,K) right=NDExt(K,rightLd)
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmLeft)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.leftLd, tiling.k));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmRight)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.k, tiling.rightLd));
        Ssyr2kGemmKernelImpl(gmLeftTensor, gmRightTensor, gmTempTensor, tiling);
    } else {
        // trans=T: left=NDExt(N,leftLd) right=DNExt(rightLd,N)
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmLeft)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.leftLd));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmRight)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.rightLd, tiling.n));
        Ssyr2kGemmKernelImpl(gmLeftTensor, gmRightTensor, gmTempTensor, tiling);
    }
}

void ssyr2k_gemm_kernel_do(
    GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmTemp,
    const Ssyr2kGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyr2k_gemm_kernel<<<numBlocks, nullptr, stream>>>(
        gmLeft, gmRight, gmTemp, tiling);
}

// ================================================================================
// Phase 2: Scale Kernel (AIV-only, SIMD)
// C = alpha * (temp1 + temp2) + beta * C (column-major C, row-major temp1/temp2)
// Only updates the uplo triangle (UPPER: j>=i, LOWER: j<=i).
// k=0 or alpha=0 also only updates uplo triangle (no temp read, C=beta*C).
// Processes 64x64 blocks; skips blocks entirely outside uplo triangle.
//
// Transpose elimination via temp1+temp2 symmetry:
//   temp1 + temp2 = op(A)*op(B)^T + op(B)*op(A)^T is symmetric, so
//   (temp1+temp2)[i][j] = (temp1+temp2)[j][i].
//   Instead of reading temp[iBase..iBase+rows][jBase..jBase+cols] (row-major -> needs
//   Gather transpose), we read temp[jBase..jBase+cols][iBase..iBase+rows] with swapped
//   coordinates. By symmetry, (temp1+temp2)[jBase+c][iBase+r] = (temp1+temp2)[iBase+r][jBase+c],
//   which is exactly the value we need. We read temp1 and temp2 separately with swapped
//   coordinates, add them in UB, and the result is in column-major layout matching cInUb.
// ================================================================================

constexpr uint32_t SSYR2K_SCALE_BLOCK = SSYR2K_ARCH35_SCALE_BLOCK;
constexpr uint32_t SSYR2K_SCALE_UB_FLOATS = SSYR2K_SCALE_BLOCK * SSYR2K_SCALE_BLOCK;

class Ssyr2kScaleAIV {
public:
    __aicore__ inline explicit Ssyr2kScaleAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC, const Ssyr2kScaleTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    Ssyr2kScaleTilingData tiling_;
    GlobalTensor<float> temp1GM_;
    GlobalTensor<float> temp2GM_;
    GlobalTensor<float> cGM_;
    TBuf<TPosition::VECIN> temp1Buf_;
    TBuf<TPosition::VECIN> temp2Buf_;
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

__aicore__ inline void Ssyr2kScaleAIV::Init(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC, const Ssyr2kScaleTilingData& tiling)
{
    tiling_ = tiling;
    temp1GM_.SetGlobalBuffer((__gm__ float*)gmTemp1);
    temp2GM_.SetGlobalBuffer((__gm__ float*)gmTemp2);
    cGM_.SetGlobalBuffer((__gm__ float*)gmC);

    pipe_.InitBuffer(temp1Buf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(temp2Buf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cInBuf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cOutBuf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));

    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_.rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_.rowsPerCore, tiling_.n);
}

__aicore__ inline void Ssyr2kScaleAIV::ApplyScale(
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

__aicore__ inline void Ssyr2kScaleAIV::ComputeScaleResult(
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

__aicore__ inline void Ssyr2kScaleAIV::ProcessBlock(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = tiling_.alphaVal;
    float beta = tiling_.betaVal;
    bool skipTemp = (tiling_.isAlphaZero || tiling_.isKZero);
    bool isBetaZero = tiling_.isBetaZero;

    uint32_t ldc = tiling_.ldc;
    uint32_t tempRowStride = tiling_.tempRowStride;
    uint32_t ubColStride = RoundUp<uint32_t>(rows, SSYR2K_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> cUb = cOutBuf_.Get<float>();
    LocalTensor<float> cInUb = cInBuf_.Get<float>();

    // Read C columns [jBase, jBase+cols), rows [iBase, iBase+rows) -- column-major, contiguous
    int64_t cSrcStride = static_cast<int64_t>(ldc - rows) * sizeof(float);
    DataCopyExtParams cpC{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), cSrcStride, 0, 0};
    DataCopyPadExtParams<float> ppC{true, 0, 0, 0.0f};
    int64_t cGmOffset = static_cast<int64_t>(jBase) * ldc + iBase;
    DataCopyPad(cInUb, cGM_[cGmOffset], cpC, ppC);

    LocalTensor<float> temp1Ub = temp1Buf_.Get<float>();

    if (!skipTemp) {
        // Read temp1 and temp2 with swapped coordinates: rows [jBase, jBase+cols), cols [iBase, iBase+rows).
        // By (temp1+temp2) symmetry: (temp1+temp2)[jBase+c][iBase+r] = (temp1+temp2)[iBase+r][jBase+c]
        // -- correct value, and DataCopyPad produces column-major UB layout matching cInUb.
        LocalTensor<float> temp2Ub = temp2Buf_.Get<float>();
        int64_t tempSrcStride = static_cast<int64_t>(tempRowStride - rows) * sizeof(float);
        DataCopyExtParams cpTemp{static_cast<uint16_t>(cols),
            static_cast<uint32_t>(rows * sizeof(float)),
            tempSrcStride, 0, 0};
        DataCopyPadExtParams<float> ppTemp{true, 0, 0, 0.0f};
        int64_t tempGmOffset = static_cast<int64_t>(jBase) * tempRowStride + iBase;
        DataCopyPad(temp1Ub, temp1GM_[tempGmOffset], cpTemp, ppTemp);
        DataCopyPad(temp2Ub, temp2GM_[tempGmOffset], cpTemp, ppTemp);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        // temp1Ub = temp1 + temp2 (in-place add, full block)
        Add(temp1Ub, temp1Ub, temp2Ub, static_cast<int32_t>(ubColStride * cols));
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    ComputeScaleResult(iBase, jBase, rows, cols, ubColStride, cUb, cInUb,
        temp1Ub, skipTemp, isBetaZero, alpha, beta);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    DataCopyExtParams cpOut{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), 0,
        cSrcStride, 0};
    DataCopyPad(cGM_[cGmOffset], cUb, cpOut);
}

__aicore__ inline void Ssyr2kScaleAIV::Process()
{
    if (rowStart_ >= rowEnd_) {
        return;
    }
    uint32_t n = tiling_.n;
    bool uploUpper = (tiling_.uploMode == ACLBLAS_UPPER);

    for (uint32_t iBase = rowStart_; iBase < rowEnd_; iBase += SSYR2K_SCALE_BLOCK) {
        uint32_t rows = Min<uint32_t>(SSYR2K_SCALE_BLOCK, rowEnd_ - iBase);
        uint32_t iEnd = iBase + rows - 1;

        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? n : Min(iEnd + 1, n);

        for (uint32_t jBase = jStart; jBase < jLimit; jBase += SSYR2K_SCALE_BLOCK) {
            uint32_t cols = Min<uint32_t>(SSYR2K_SCALE_BLOCK, jLimit - jBase);
            ProcessBlock(iBase, jBase, rows, cols);
        }
    }
}

extern "C" __global__ __aicore__ void ssyr2k_scale_kernel(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC,
    const Ssyr2kScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Ssyr2kScaleAIV op(pipe);
    op.Init(gmTemp1, gmTemp2, gmC, tiling);
    op.Process();
}

void ssyr2k_scale_kernel_do(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC,
    const Ssyr2kScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyr2k_scale_kernel<<<numBlocks, nullptr, stream>>>(
        gmTemp1, gmTemp2, gmC, tiling);
}
