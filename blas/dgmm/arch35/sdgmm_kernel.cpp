/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sdgmm_kernel.cpp
 * \brief Device-side kernel for aclblasSdgmm (arch35).
 *        Pure tensor_api (AscendC::Te): MakeTensor / MakeMemPtr / MakeFrameLayout /
 *        Copy(CopyGM2UB/CopyUB2GM) / Transform<Inst::Mul|Inst::MulScalar>.
 *        Column-major storage (BLAS convention).
 *
 *        mode=R: per-column scalar broadcast multiply (Transform<MulScalar>).
 *        mode=L: per-row-segment vector elementwise multiply (Transform<Mul>),
 *                x segment resident in UB and reused across columns.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "sdgmm_kernel.h"

using namespace AscendC;
using namespace AscendC::Te;

// ==========================================================================
// mode=R: per-column scalar broadcast multiply.
// For each column j: read scalar x[j*incx], then Transform<MulScalar> to
// scale the A column segment. x values are batch-copied to UB once per
// core to reduce GM scalar reads.
// ==========================================================================
template <typename XGM, typename AGM, typename CGM>
__aicore__ inline void SdgmmProcessRight(
    XGM xGm, AGM aGm, CGM cGm,
    const SdgmmTilingData& tiling,
    uint32_t startCol, uint32_t endCol,
    uint32_t startM, uint32_t endM,
    uint32_t ubOffsetA, uint32_t ubOffsetC, uint32_t ubOffsetX)
{
    // UB tensors: 1D contiguous buffers (NDExt with 1 row).
    auto ubA = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetA),
        MakeFrameLayout<NDExtLayoutPtn>(1u, tiling.tileM));
    auto ubC = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetC),
        MakeFrameLayout<NDExtLayoutPtn>(1u, tiling.tileM));

    auto copyGM2UB = MakeCopy(CopyGM2UB{});
    auto copyUB2GM = MakeCopy(CopyUB2GM{});

    int64_t absIncx = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                         : -static_cast<int64_t>(tiling.incx);

    // Batch-copy the x segment for [startCol, endCol) into UB.
    // incx==1: contiguous copy x[startCol..endCol-1].
    // incx!=1: copy the contiguous GM range covering all needed x indices,
    //          then index into UB with stride. The GM range is:
    //   incx>0: [startCol*incx, (endCol-1)*incx]
    //   incx<0: [(n-1-(endCol-1))*|incx|, (n-1-startCol)*|incx|]
    uint32_t colCount = endCol - startCol;
    uint64_t xSegSize = (colCount <= 1) ? 1
        : static_cast<uint64_t>(colCount - 1) * static_cast<uint64_t>(absIncx) + 1;
    auto ubX = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetX),
        MakeFrameLayout<NDExtLayoutPtn>(1u, xSegSize));

    uint64_t xGmStart;
    if (tiling.incx >= 0) {
        xGmStart = static_cast<uint64_t>(startCol) * static_cast<uint64_t>(tiling.incx);
    } else {
        xGmStart = static_cast<uint64_t>(tiling.n - 1 - (endCol - 1)) * static_cast<uint64_t>(absIncx);
    }
    auto gmXSeg = xGm.Slice(MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(xGmStart)),
                             MakeShape(1u, xSegSize));
    Copy(copyGM2UB, ubX, gmXSeg);
    PipeBarrier<PIPE_MTE2>();

    for (uint32_t j = startCol; j < endCol; j++) {
        // Read scalar x[k] from UB (avoiding per-column GM scalar load).
        uint64_t ubIdx;
        if (tiling.incx >= 0) {
            ubIdx = static_cast<uint64_t>(j - startCol) * static_cast<uint64_t>(tiling.incx);
        } else {
            ubIdx = static_cast<uint64_t>(tiling.n - 1 - j - (tiling.n - 1 - (endCol - 1))) * static_cast<uint64_t>(absIncx);
        }
        float xj = ubX[MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(ubIdx))];

        uint32_t rowOffset = startM;
        for (uint32_t remain = endM - startM; remain > 0;) {
            uint32_t curM = (remain > tiling.tileM) ? tiling.tileM : remain;

            auto gmACol = aGm.Slice(MakeCoord(j, rowOffset), MakeShape(1u, curM));
            auto ubASlice = ubA.Slice(MakeCoord(0u, 0u), MakeShape(1u, curM));

            Copy(copyGM2UB, ubASlice, gmACol);
            PipeBarrier<PIPE_MTE2>();

            auto ubCSlice = ubC.Slice(MakeCoord(0u, 0u), MakeShape(1u, curM));
            Transform<Inst::MulScalar>(ubCSlice, ubASlice, xj);
            PipeBarrier<PIPE_V>();

            auto gmCCol = cGm.Slice(MakeCoord(j, rowOffset), MakeShape(1u, curM));
            Copy(copyUB2GM, gmCCol, ubCSlice);
            PipeBarrier<PIPE_MTE3>();

            rowOffset += curM;
            remain -= curM;
        }
    }
}

// ==========================================================================
// mode=L: per-row-segment vector elementwise multiply.
// Outer loop over m row-segments: load the x segment once into UB and reuse
// it for all columns owned by this core.
// incx==1: contiguous Copy GM→UB (fast path).
// incx!=1: strided scalar load via tensor_api operator[] loop.
// ==========================================================================
template <typename XGM, typename AGM, typename CGM>
__aicore__ inline void SdgmmProcessLeft(
    XGM xGm, AGM aGm, CGM cGm,
    const SdgmmTilingData& tiling,
    uint32_t startCol, uint32_t endCol,
    uint32_t startM, uint32_t endM,
    uint32_t ubOffsetA, uint32_t ubOffsetC, uint32_t ubOffsetX)
{
    auto ubA = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetA),
        MakeFrameLayout<NDExtLayoutPtn>(1u, tiling.tileM));
    auto ubC = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetC),
        MakeFrameLayout<NDExtLayoutPtn>(1u, tiling.tileM));
    auto ubX = MakeTensor(
        MakeMemPtr<Location::UB, float>(ubOffsetX),
        MakeFrameLayout<NDExtLayoutPtn>(1u, tiling.tileM));

    auto copyGM2UB = MakeCopy(CopyGM2UB{});
    auto copyUB2GM = MakeCopy(CopyUB2GM{});

    int64_t absIncxL = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                           : -static_cast<int64_t>(tiling.incx);

    uint32_t rowOffset = startM;
    for (uint32_t remain = endM - startM; remain > 0;) {
        uint32_t curM = (remain > tiling.tileM) ? tiling.tileM : remain;

        // Load x row-segment into UB.
        auto ubXSlice = ubX.Slice(MakeCoord(0u, 0u), MakeShape(1u, curM));
        if (tiling.incx == 1) {
            // Contiguous: use Copy GM→UB.
            auto gmXSeg = xGm.Slice(MakeCoord(0u, rowOffset), MakeShape(1u, curM));
            Copy(copyGM2UB, ubXSlice, gmXSeg);
            PipeBarrier<PIPE_MTE2>();
        } else {
            // Strided: load element by element via tensor_api operator[].
            // incx>=0: index = (rowOffset+i)*incx;
            // incx<0:  index = (m-1-(rowOffset+i))*|incx| (BLAS reverse).
            // NOTE: incx!=1 且 m 较大时有显著性能退化（逐元素 GM scalar 读 +
            // UB scalar 写）。tensor_api 的 Copy(CopyGM2UB) 不支持 strided
            // GM 源，后续可考虑 dedicated gather kernel 或重排 x 布局。
            for (uint32_t i = 0; i < curM; i++) {
                int64_t logicalIdx = static_cast<int64_t>(rowOffset + i);
                int64_t xOffset = (tiling.incx >= 0)
                    ? logicalIdx * tiling.incx
                    : (static_cast<int64_t>(tiling.m - 1) - logicalIdx) * absIncxL;
                ubXSlice[MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(i))] =
                    xGm[MakeCoord(static_cast<int64_t>(0), xOffset)];
            }
            PipeBarrier<PIPE_V>();
        }

        // Inner loop over columns: reuse ubXSlice for every column.
        for (uint32_t j = startCol; j < endCol; j++) {
            auto gmACol = aGm.Slice(MakeCoord(j, rowOffset), MakeShape(1u, curM));
            auto ubASlice = ubA.Slice(MakeCoord(0u, 0u), MakeShape(1u, curM));

            Copy(copyGM2UB, ubASlice, gmACol);
            PipeBarrier<PIPE_MTE2>();

            // Compute: ubC = ubA * ubX (elementwise)
            auto ubCSlice = ubC.Slice(MakeCoord(0u, 0u), MakeShape(1u, curM));
            Transform<Inst::Mul>(ubCSlice, ubASlice, ubXSlice);
            PipeBarrier<PIPE_V>();

            auto gmCCol = cGm.Slice(MakeCoord(j, rowOffset), MakeShape(1u, curM));
            Copy(copyUB2GM, gmCCol, ubCSlice);
            PipeBarrier<PIPE_MTE3>();
        }

        rowOffset += curM;
        remain -= curM;
    }
}

// ==========================================================================
// Kernel entry — pure tensor_api, handles all mode/incx combinations.
// 2D block decomposition: blockIdx = colBlock * mBlocks + mBlock.
// ==========================================================================
extern "C" __global__ __aicore__ void sdgmm_aiv_kernel(
    GM_ADDR x, GM_ADDR A, GM_ADDR C, const SdgmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockIdx = GetBlockIdx();
    uint32_t mBlocks = tiling.mBlocks;
    if (mBlocks == 0) {
        mBlocks = 1;
    }

    // 2D decomposition: colBlock × mBlock
    uint32_t colBlock = blockIdx / mBlocks;
    uint32_t mBlock = blockIdx % mBlocks;
    uint32_t numColBlocks = GetBlockNum() / mBlocks;

    // Balanced column distribution: first `remainder` blocks get perCoreN+1,
    // the rest get perCoreN.
    uint32_t startCol, endCol;
    if (colBlock < tiling.remainder) {
        startCol = colBlock * (tiling.perCoreN + 1);
        endCol = startCol + tiling.perCoreN + 1;
    } else {
        startCol = colBlock * tiling.perCoreN + tiling.remainder;
        endCol = startCol + tiling.perCoreN;
    }

    // Balanced m-tile distribution: first `mTileRemainder` blocks get
    // perCoreMTile+1 tiles, the rest get perCoreMTile.
    uint32_t startMTile, endMTile;
    if (mBlock < tiling.mTileRemainder) {
        startMTile = mBlock * (tiling.perCoreMTile + 1);
        endMTile = startMTile + tiling.perCoreMTile + 1;
    } else {
        startMTile = mBlock * tiling.perCoreMTile + tiling.mTileRemainder;
        endMTile = startMTile + tiling.perCoreMTile;
    }
    uint32_t startM = startMTile * tiling.tileM;
    uint32_t endM = endMTile * tiling.tileM;
    if (endM > tiling.m) {
        endM = tiling.m;
    }

    if (tiling.m == 0 || tiling.n == 0 || tiling.tileM == 0 ||
        startCol >= endCol || startM >= endM) {
        return;
    }

    // Create GM tensors.
    // NDExtLayoutPtn(n, lda): element [j, i] = j*lda + i = A[i,j] (column-major).
    // x is 1D: NDExtLayoutPtn(1, xTotalEl) where xTotalEl covers the full strided
    // storage (BLAS convention: 1 + (xLen-1)*|incx| elements).
    uint32_t xLen = (tiling.mode == SDGMM_MODE_LEFT) ? tiling.m : tiling.n;
    int64_t absIncx = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                         : -static_cast<int64_t>(tiling.incx);
    uint64_t xTotalEl = static_cast<uint64_t>(xLen - 1) * static_cast<uint64_t>(absIncx) + 1;
    auto xGm = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(x)),
        MakeFrameLayout<NDExtLayoutPtn>(1u, xTotalEl));
    auto aGm = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(A)),
        MakeFrameLayout<NDExtLayoutPtn>(static_cast<uint64_t>(tiling.n),
                                        static_cast<uint64_t>(tiling.lda)));
    auto cGm = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(C)),
        MakeFrameLayout<NDExtLayoutPtn>(static_cast<uint64_t>(tiling.n),
                                        static_cast<uint64_t>(tiling.ldc)));

    // UB buffer offsets (byte offsets from UB base).
    // tileM is aligned to 8 (ALIGN_UNIT), so tileM * sizeof(float) is 32-byte aligned.
    uint32_t ubOffsetA = 0;
    uint32_t ubOffsetC = tiling.tileM * sizeof(float);
    uint32_t ubOffsetX = ubOffsetC + tiling.tileM * sizeof(float);

    if (tiling.mode == SDGMM_MODE_RIGHT) {
        SdgmmProcessRight(xGm, aGm, cGm, tiling, startCol, endCol,
                          startM, endM, ubOffsetA, ubOffsetC, ubOffsetX);
    } else {
        SdgmmProcessLeft(xGm, aGm, cGm, tiling, startCol, endCol,
                         startM, endM, ubOffsetA, ubOffsetC, ubOffsetX);
    }
}

// Kernel launcher: asynchronously launches the tensor_api kernel.
// tiling.mode is the normalized value (SDGMM_MODE_LEFT / SDGMM_MODE_RIGHT).
void sdgmm_kernel_do(GM_ADDR x, GM_ADDR A, GM_ADDR C,
                     const SdgmmTilingData& tiling,
                     uint32_t numBlocks, void* stream)
{
    sdgmm_aiv_kernel<<<numBlocks, nullptr, stream>>>(x, A, C, tiling);
}
