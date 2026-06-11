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
 * \file aclblas_kernel_do.h
 * \brief ops-blas kernel_do 内部kernel入口函数(不对外暴露)
 */

#include <cstdint>

void cscal_kernel_do(uint8_t* x, uint8_t* maskBuf, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

struct SscalTilingData;

// arch35-style: tiling passed by value
void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, const SscalTilingData& tiling,
                     uint32_t numBlocks, void *stream);

// arch22-style: tiling passed as GM pointer (for backward compatibility)
void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void caxpy_kernel_do(uint8_t* x, uint8_t* maskBuf, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void cgemm_kernel_do(uint8_t* d_A, uint8_t* d_B, uint8_t* d_A_r, uint8_t* d_A_i,
                     uint8_t* d_B_r, uint8_t* d_B_i, uint8_t* d_C_rr, uint8_t* d_C_ri,
                     uint8_t* d_C_ir, uint8_t* d_C_ii, uint8_t* d_C, uint8_t* workspace,
                     uint8_t* tilingGm, uint32_t numBlocks, void *stream);

void cgemm_batched_kernel_do(uint8_t* a, uint8_t* b, uint8_t* gatherOffset, uint8_t* c,
                             uint8_t* workSpace, uint8_t* tilingGm,
                             uint32_t numBlocks, void *stream);

void cgemv_no_trans_kernel_do(uint8_t* d_A, uint8_t* d_x, uint8_t* d_y_in, uint8_t* maskBuf, uint8_t* d_y,
                              uint8_t* workSpace, uint8_t* tilingGm,
                              uint32_t numBlocks, void *stream);

void cgemv_do_trans_kernel_do(uint8_t* d_A, uint8_t* d_x, uint8_t* d_y_in, uint8_t* maskBuf, uint8_t* d_y,
                              uint8_t* workSpace, uint8_t* tilingGm,
                              uint32_t numBlocks, void *stream);

void cgemv_batched_kernel_do(uint8_t* A, uint8_t* x, uint8_t* mask, uint8_t* y,
                             uint8_t* workSpace, uint8_t* tilingGm,
                             uint32_t numBlocks, void *stream);

void gemv_batched_kernel_do(uint8_t* A, uint8_t* x, uint8_t* y,
                            uint8_t* workSpace, uint8_t* tilingGm,
                            uint32_t numBlocks, void *stream);

void cgerc_kernel_do(uint8_t* d_x, uint8_t* d_y, uint8_t* d_offset, uint8_t* d_A, uint8_t* work_space,
                     uint8_t* tiling_gm, uint32_t num_blocks, void *stream);

void colwise_mul_kernel_do(uint8_t* mat, uint8_t* vec, uint8_t* aug, uint8_t* result,
                           uint8_t* workSpace, uint8_t* tilingGm,
                           uint32_t numBlocks, void *stream);

void complex_mat_dot_kernel_do(uint8_t* matx, uint8_t* maty, uint8_t* aug, uint8_t* result,
                               uint8_t* tilingGm, uint32_t numBlocks, void *stream);

void scopy_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void ctrmv_kernel_do(uint8_t* gm_A, uint8_t* gm_X, uint8_t* gm_uplo,
                     uint8_t* gm_wksp, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void cdot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

void sdot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

void iamax_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling,
                     uint32_t numBlocks, void *stream);

void snrm2_kernel_do(uint8_t* x, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void csrot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

struct SasumTilingData;

// arch35-style: tiling passed by value
void sasum_kernel_do(uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace,
                     const SasumTilingData& tiling, uint32_t numBlocks, void *stream);

// arch22-style: tiling passed as GM pointer (for backward compatibility)
void sasum_kernel_do(uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void ssyr_kernel_do(uint8_t* gm_x, uint8_t* gm_A, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

struct SyrTilingData;

void syr_kernel_do(uint8_t* x, uint8_t* A, const SyrTilingData &tiling,
                   uint32_t numBlocks, void *stream);

void ssyr2_kernel_do(uint8_t* gm_x, uint8_t* gm_y, uint8_t* gm_A, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

struct Syr2TilingData;

void syr2_kernel_do(uint8_t* x, uint8_t* y, uint8_t* A, const Syr2TilingData &tiling,
                     uint32_t numBlocks, void *stream);

void strmm_kernel_do(uint8_t* d_A, uint8_t* d_B, uint8_t* d_C, uint8_t* workspace,
                     uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void strmv_kernel_do(uint8_t* gm_A, uint8_t* gm_X, uint8_t* gm_uplo, uint8_t* gm_output,
                     uint8_t* gm_wksp, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void tbmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

void tpmv_kernel_do(uint8_t* aPacked, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

void sswap_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void cswap_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void stpttr_kernel_do(uint8_t* aPacked, uint8_t* aFull, uint8_t* workSpace,
                      uint8_t* tilingGm, uint32_t numBlocks, void *stream);

void strttp_kernel_do(uint8_t* a, uint8_t* ap, uint8_t* tilingGm,
                      uint32_t numBlocks, void *stream);

void ssbmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                      uint32_t numBlocks, void *stream);

void sgbmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                      uint32_t numBlocks, void *stream);

void ssymv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                      uint32_t numBlocks, void *stream);

void spmv_kernel_do(uint8_t* aPacked, uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* workSpace, uint8_t* tilingGm,
                    uint32_t numBlocks, void *stream);

struct SgerTilingData;

void sger_kernel_do(uint8_t* A, uint8_t* x, uint8_t* y, uint8_t* alpha, uint8_t* tilingGm,
                   uint32_t numBlocks, void *stream);

void sger_arch35_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* A, const SgerTilingData &tiling, uint32_t numBlocks, void *stream);

struct SrotmTilingData;

void srotm_kernel_do(const SrotmTilingData &tiling, uint32_t numBlocks, void *stream);

struct StpsvTilingData;

void stpsv_kernel_do(const StpsvTilingData &tiling, void *stream);

void sspmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

void sgemv_kernel_do(uint8_t* A, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);
struct StrsvTilingData;

void strsv_kernel_do(uint8_t* gmAddrA, uint8_t* gmAddrX, const StrsvTilingData &tiling, void *stream);

void sgeqrf_batched_kernel_do(uint8_t* aarray, uint8_t* tauArray, uint8_t* tilingGm,
                             uint32_t numBlocks, void *stream);
struct SgetrfBatchedTilingData;

void sgetrf_batched_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray,
    const SgetrfBatchedTilingData &tiling, uint32_t numBlocks, void *stream);

struct SgelsBatchedTilingData;

void sgels_decompose_kernel_do(
    uint8_t* aArray, uint8_t* cArray, uint8_t* workspace, uint8_t* devInfo,
    const SgelsBatchedTilingData &tiling, uint32_t numBlocks, void *stream);
void srotm_kernel_do_arch35(
    uint8_t* x, uint8_t* y, const SrotmTilingData& tilingData, uint32_t numBlocks, void *stream);
