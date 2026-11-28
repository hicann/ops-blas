/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_regular_host.h"
#include "epilogue_alpha_beta_host.h"
#include "epilogue_alpha_beta_tiling_data.h"
#include "host_utils.h"

#include <algorithm>
#include <limits>

namespace {
constexpr uint64_t INT8_ALIGNMENT = 4;
constexpr uint64_t MAX_INT8_K = std::numeric_limits<int32_t>::max() / (128 * 128);

struct TileSize {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t c0;
};

TileSize GetTileSize(LtRegularMatmulType type)
{
    switch (type) {
        case LtRegularMatmulType::FP16:
        case LtRegularMatmulType::BF16:
            return {128, 128, 16, 16};
        case LtRegularMatmulType::INT8:
            return {128, 128, 32, 32};
        default:
            return {32, 16, 32, 32};
    }
}

GemmStridedBatchedExTilingData BuildRegularTiling(
    const MatmulProblem& p, LtRegularMatmulType type, uint32_t lda, uint32_t ldb, bool directOutput)
{
    const TileSize tile = GetTileSize(type);
    GemmStridedBatchedExTilingData t{};
    t.m = static_cast<int32_t>(p.m);
    t.n = static_cast<int32_t>(p.n);
    t.k = static_cast<int32_t>(p.k);
    t.lda = static_cast<int32_t>(lda);
    t.ldb = static_cast<int32_t>(ldb);
    t.ldc = directOutput ? static_cast<int32_t>(p.ldd) : t.n;
    t.logicalM = t.m;
    t.logicalN = t.n;
    t.baseM = t.singleCoreM = static_cast<int32_t>(tile.m);
    t.baseN = t.singleCoreN = static_cast<int32_t>(tile.n);
    t.baseK = static_cast<int32_t>(tile.k);
    t.c0Size = static_cast<int32_t>(tile.c0);
    t.mBlocks = CeilDiv<int32_t>(p.m, tile.m);
    t.nBlocks = CeilDiv<int32_t>(p.n, tile.n);
    t.totalTasks = static_cast<uint64_t>(t.mBlocks) * t.nBlocks;
    t.usedCoreNum = static_cast<int32_t>(std::min<uint64_t>(p.numBlocks, t.totalTasks));
    t.isTransA = p.transA;
    t.isTransB = p.transB;
    t.batchCount = 1;
    t.splitK = 1;
    t.selectedAlgo = GEMM_STRIDED_KERNEL_BALANCED;
    return t;
}

uint8_t* AsBytes(const void* address) { return static_cast<uint8_t*>(const_cast<void*>(address)); }

void LaunchInt32Epilogue(const MatmulProblem& p, uint8_t* raw)
{
    EpilogueAlphaBetaTilingData grid;
    epilogue_alpha_beta_get_tiling(static_cast<uint32_t>(p.m), static_cast<uint32_t>(p.n), p.numBlocks, grid);
    const LtInt32EpilogueTiling t{
        grid.m, grid.n, p.ldc, p.ldd, grid.usedCoreNum, grid.numThreads, p.alphaInt, p.betaInt};
    matmul_int32_epilogue_do(raw, AsBytes(p.C), AsBytes(p.D), t, p.stream);
}

aclblasStatus_t LaunchRegularKernels(
    const MatmulProblem& p, LtRegularMatmulType type, uint32_t lda, uint32_t ldb, bool directOutput)
{
    const auto tiling = BuildRegularTiling(p, type, lda, ldb, directOutput);
    uint8_t* output = AsBytes(directOutput ? p.D : p.workspace);
    matmul_regular_cube_do(AsBytes(p.A), AsBytes(p.B), output, tiling, type, directOutput, p.stream);
    if (directOutput) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (type == LtRegularMatmulType::INT8) {
        LaunchInt32Epilogue(p, output);
    } else {
        epilogue_alpha_beta_do(output, AsBytes(p.C), AsBytes(p.D), static_cast<uint32_t>(p.m),
            static_cast<uint32_t>(p.n), p.ldc, p.ldd, static_cast<uint32_t>(p.n), p.alpha, p.beta, p.dtypeC,
            ACL_FLOAT, p.dtypeD, p.stream, nullptr, nullptr);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

bool CanTileRegularMatmul(
    LtRegularMatmulType type, uint64_t m, uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
    uint64_t ldc, uint64_t ldd)
{
    if (k == 0 || std::max({m, n, k, lda, ldb, ldc, ldd}) > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    if (type == LtRegularMatmulType::INT8 && (k > MAX_INT8_K || (lda | ldb) % INT8_ALIGNMENT != 0)) {
        return false;
    }
    const TileSize tile = GetTileSize(type);
    return CeilDivU64(m, tile.m) * CeilDivU64(n, tile.n) <= std::numeric_limits<uint32_t>::max();
}

size_t RegularMatmulWorkspaceSize(uint64_t m, uint64_t n)
{
    return static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
}

bool CanTileRegularMatmulLayouts(
    LtRegularMatmulType type, const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a,
    const aclblasLtMatrixLayoutImpl* b, const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d)
{
    const uint64_t k = desc->transA == ACLBLAS_OP_N ? a->cols : a->rows;
    return CanTileRegularMatmul(type, d->rows, d->cols, k,
        a->ld != 0 ? static_cast<uint64_t>(a->ld) : a->cols,
        b->ld != 0 ? static_cast<uint64_t>(b->ld) : b->cols,
        c->ld != 0 ? static_cast<uint64_t>(c->ld) : d->cols,
        d->ld != 0 ? static_cast<uint64_t>(d->ld) : d->cols);
}

aclblasStatus_t MatmulRegularLaunch(const MatmulProblem& p, LtRegularMatmulType type)
{
    const uint32_t lda = p.lda != 0 ? p.lda : static_cast<uint32_t>(p.transA ? p.m : p.k);
    const uint32_t ldb = p.ldb != 0 ? p.ldb : static_cast<uint32_t>(p.transB ? p.k : p.n);
    if (!CanTileRegularMatmul(type, p.m, p.n, p.k, lda, ldb, p.ldc, p.ldd)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool integer = type == LtRegularMatmulType::INT8;
    if (integer && (reinterpret_cast<uintptr_t>(p.A) | reinterpret_cast<uintptr_t>(p.B)) % INT8_ALIGNMENT != 0) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool useC = integer ? p.betaInt != 0 : p.beta != 0.0f;
    if (useC && (p.C == nullptr || (p.C == p.D && p.ldc != p.ldd))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const bool unitAlpha = integer ? p.alphaInt == 1 : p.alpha == 1.0f;
    const bool directOutput = unitAlpha && !useC;
    if (!directOutput && (p.workspace == nullptr || p.workspaceSize < RegularMatmulWorkspaceSize(p.m, p.n))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return LaunchRegularKernels(p, type, lda, ldb, directOutput);
}
