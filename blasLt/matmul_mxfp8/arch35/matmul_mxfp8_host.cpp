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
 * \file matmul_mxfp8_host.cpp
 * \brief Host-side MXFP8 tiling helper.
 */

#include "version/asc_devkit_version.h"

#if ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0

#include <algorithm>
#include <cstdint>

#include "matmul_mxfp8_host.h"
#include "quant_matmul_tiling_data.h"

void matmul_mxfp8_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, QuantMatmulTilingData& tilingData)
{
    (void)transA;
    (void)transB;
    tilingData = {};
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);

    constexpr uint32_t kAlignMx = 32U;
    tilingData.baseM = static_cast<uint32_t>(std::min<uint64_t>(m, 256ULL));
    tilingData.baseN = static_cast<uint32_t>(std::min<uint64_t>(n, 256ULL));
    tilingData.baseK = static_cast<uint32_t>(std::min<uint64_t>(k, 128ULL));
    if (tilingData.baseK % kAlignMx != 0U) {
        tilingData.baseK = static_cast<uint32_t>((tilingData.baseK + kAlignMx - 1U) / kAlignMx * kAlignMx);
        tilingData.baseK = static_cast<uint32_t>(std::min<uint64_t>(tilingData.baseK, k));
    }

    tilingData.scaleKL1 = std::min(tilingData.baseK, tilingData.k);
    tilingData.stepK = 4U;
    tilingData.nBufferNum = 4U;
    tilingData.dbL0c = 2U;
    tilingData.mTailTile = 1U;
    tilingData.nTailTile = 1U;
    tilingData.mBaseTailSplitCnt = 1U;
    tilingData.nBaseTailSplitCnt = 1U;
    tilingData.mTailMain = tilingData.m;
    tilingData.nTailMain = tilingData.n;
    tilingData.usedCoreNum = 24U;
}

#endif // ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
