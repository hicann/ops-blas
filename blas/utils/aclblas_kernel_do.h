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

#define GM_ADDR uint8_t*

void cscal_kernel_do(GM_ADDR x, GM_ADDR maskBuf, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void sscal_kernel_do(GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void caxpy_kernel_do(GM_ADDR x, GM_ADDR maskBuf, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void cgemm_kernel_do(GM_ADDR d_A, GM_ADDR d_B, GM_ADDR d_A_r, GM_ADDR d_A_i,
                     GM_ADDR d_B_r, GM_ADDR d_B_i, GM_ADDR d_C_rr, GM_ADDR d_C_ri,
                     GM_ADDR d_C_ir, GM_ADDR d_C_ii, GM_ADDR d_C, GM_ADDR workspace,
                     GM_ADDR tilingGm, uint32_t numBlocks, void *stream);

void cgemm_batched_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR gatherOffset, GM_ADDR c,
                             GM_ADDR workSpace, GM_ADDR tilingGm,
                             uint32_t numBlocks, void *stream);

void cgemv_no_trans_kernel_do(GM_ADDR d_A, GM_ADDR d_x, GM_ADDR d_y_in, GM_ADDR maskBuf, GM_ADDR d_y,
                              GM_ADDR workSpace, GM_ADDR tilingGm,
                              uint32_t numBlocks, void *stream);

void cgemv_do_trans_kernel_do(GM_ADDR d_A, GM_ADDR d_x, GM_ADDR d_y_in, GM_ADDR maskBuf, GM_ADDR d_y,
                              GM_ADDR workSpace, GM_ADDR tilingGm,
                              uint32_t numBlocks, void *stream);

void cgemv_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                             GM_ADDR workSpace, GM_ADDR tilingGm,
                             uint32_t numBlocks, void *stream);

void cgerc_kernel_do(GM_ADDR d_x, GM_ADDR d_y, GM_ADDR d_offset, GM_ADDR d_A, GM_ADDR work_space,
                     GM_ADDR tiling_gm, uint32_t num_blocks, void *stream);

void colwise_mul_kernel_do(GM_ADDR mat, GM_ADDR vec, GM_ADDR aug, GM_ADDR result,
                           GM_ADDR workSpace, GM_ADDR tilingGm,
                           uint32_t numBlocks, void *stream);

void complex_mat_dot_kernel_do(GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result,
                               GM_ADDR tilingGm, uint32_t numBlocks, void *stream);

void scopy_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void ctrmv_kernel_do(GM_ADDR gm_A, GM_ADDR gm_X, GM_ADDR gm_uplo,
                     GM_ADDR gm_wksp, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void cdot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm,
                    uint32_t numBlocks, void *stream);

void sdot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm,
                    uint32_t numBlocks, void *stream);

void iamax_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                     uint32_t numBlocks, void *stream);

void snrm2_kernel_do(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void csrot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void sasum_kernel_do(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void ssyr_kernel_do(GM_ADDR gm_x, GM_ADDR gm_A, GM_ADDR workSpace, GM_ADDR tilingGm,
                    uint32_t numBlocks, void *stream);

void ssyr2_kernel_do(GM_ADDR gm_x, GM_ADDR gm_y, GM_ADDR gm_A, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void strmm_kernel_do(GM_ADDR d_A, GM_ADDR d_B, GM_ADDR d_C, GM_ADDR workspace,
                     GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void strmv_kernel_do(GM_ADDR gm_A, GM_ADDR gm_X, GM_ADDR gm_uplo, GM_ADDR gm_output,
                     GM_ADDR gm_wksp, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void sswap_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);

void cswap_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream);