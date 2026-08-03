/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

namespace te = AscendC::Te;

constexpr int64_t SYRK_ARCH35_L0A_SIZE = 64 * 1024;
constexpr int64_t SYRK_ARCH35_L0C_SIZE = 256 * 1024;

constexpr uint64_t SYRK_ARCH35_FP32_C0 = 8;
constexpr uint64_t SYRK_ARCH35_FRACTAL = 16;
constexpr uint64_t SYRK_ARCH35_L0C_C0 = 16;
constexpr uint64_t SYRK_ARCH35_L0_BUF_MASK = 0x1;
constexpr uint64_t SYRK_ARCH35_HALF_L0_SIZE = SYRK_ARCH35_L0A_SIZE / 2;
constexpr uint64_t SYRK_ARCH35_L0C_BUF_MASK = 0x1;
constexpr uint64_t SYRK_ARCH35_HALF_L0C_SIZE = SYRK_ARCH35_L0C_SIZE / 2;

template <typename CopyAtom, typename TensorGM, typename TensorL1>
__aicore__ inline void SyrkGemmCopyGM2L1(
    CopyAtom copyGM2L1, TensorGM gmTensor, TensorL1 tensorL1,
    uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void SyrkGemmL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0,
    uint64_t iter0, uint64_t baseK,
    uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);

    uint64_t l0cOffset = l0cBufId * SYRK_ARCH35_HALF_L0C_SIZE;
    auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curML1, nL0);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & SYRK_ARCH35_L0_BUF_MASK;
        uint64_t l0Offset = SYRK_ARCH35_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, nL0);
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
    SyrkGemmCopyGM2L1(copyGM2L1, gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    SyrkGemmCopyGM2L1(copyGM2L1, gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    SyrkGemmL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0,
        iter0, baseK, l0PingPong, l0cBufId);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorA, typename TensorB>
__aicore__ inline void SyrkGemmProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK,
    uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cBufId,
    uint32_t l1BufNum, int64_t l1Size)
{
    using T = float;
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    if (tileKChunk == 0) {
        return;
    }
    for (uint32_t kOff = 0; kOff < K; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & (l1BufNum - 1);
        uint64_t l1OffsetA = l1BufId * (l1Size / l1BufNum);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, SYRK_ARCH35_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, nL0));
        SyrkGemmProcessKChunk(copyGM2L1, gmLeftTensor, gmRightTensor,
            tensorAL1, tensorBL1, l1BufId,
            mOff, nOff, kOff, curK, mL0, nL0,
            kOff / tileKChunk, baseK, l0PingPong, l0cBufId);
        abL1LoopCnt++;
    }
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessTile(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd,
    uint32_t tileM, uint32_t tileN,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cPingPong,
    uint32_t l1BufNum, int64_t l1Size)
{
    auto copyL0C2GMAtom = te::MakeCopy(te::CopyL0C2GM{});

    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            uint64_t l0cBufId = l0cPingPong & SYRK_ARCH35_L0C_BUF_MASK;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            SyrkGemmProcessKChunks(gmLeftTensor, gmRightTensor,
                K, tileKChunk, baseK,
                mOff, nOff, curTileM, curTileN, abL1LoopCnt, l0PingPong, l0cBufId,
                l1BufNum, l1Size);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
            uint64_t l0cOffset = l0cBufId * SYRK_ARCH35_HALF_L0C_SIZE;
            auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curTileM, curTileN);
            auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
            auto gmBlockC = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curTileM, curTileN));
            copyL0C2GMAtom.Call(gmBlockC, tensorL0C, te::FixpipeParams{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            l0cPingPong++;
        }
    }
}

template <typename TensorA, typename TensorB, typename TensorTemp, typename TilingData>
__aicore__ inline void SyrkGemmKernelImpl(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    const TilingData& tiling,
    uint32_t baseK, uint32_t l1BufNum, int64_t l1Size)
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
    uint64_t blockNum = AscendC::GetBlockNum();

    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += blockNum) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }
        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        SyrkGemmProcessTile(gmLeftTensor, gmRightTensor, gmTempTensor,
            K, tiling.tileKChunk, baseK,
            mStart, Min<uint32_t>(mStart + tiling.singleCoreM, n),
            nStart, Min<uint32_t>(nStart + tiling.singleCoreN, n),
            tiling.tileM, tiling.tileN, abL1LoopCnt, l0PingPong, l0cPingPong,
            l1BufNum, l1Size);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
}
