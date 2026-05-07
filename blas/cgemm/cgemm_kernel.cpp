
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "../common/common.h"
#include "../common/iterator.h"
#include "../common/simd.h"
#include "../common/utils.h"
#include "../common/ascblas_kernel_utils.h"

// #define GM_ADDR uint8_t*

using namespace fp32;

#ifdef __DAV_C220_CUBE__
__aicore__ __inline__ void ascblasSmatmul(ascblasOperation_t transA, ascblasOperation_t transB, int64_t M, int64_t N,
                                          int64_t K, AscendC::GlobalTensor<float> gm_A, int64_t lda,
                                          AscendC::GlobalTensor<float> gm_B, int64_t ldb,
                                          AscendC::GlobalTensor<float> gm_C, int64_t ldc, int64_t batchSize, int64_t M0,
                                          int64_t N0, int64_t K0, int64_t is_AIV_AIC_PIPE,
                                          AscendC::GlobalTensor<float> AIV_AIC_workspace)
{
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> L1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    AscendC::LocalTensor<float> L1_base_b = buf.GetBuffer<BufferType::ASCEND_CB, float>(128 * 1024);

    AscendC::LocalTensor<float> L0A_base = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    AscendC::LocalTensor<float> L0B_base = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    AscendC::LocalTensor<float> L0C_base = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    if (M0 == 0) M0 = 128;
    if (N0 == 0) N0 = 128;
    if (K0 == 0) K0 = 128;

    int64_t m_loop = (M + M0 - 1) / M0;
    if (m_loop == 0) m_loop = 1;
    int64_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) n_loop = 1;
    int64_t k_loop = (K + K0 - 1) / K0;
    int64_t loop = batchSize * m_loop * n_loop;

    int64_t loop_ping_flag = 1;
    int64_t k_loop_ping_flag = 1;

    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    int64_t blockNum = AscendC::GetBlockNum();
    for (int64_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % blockNum != AscendC::GetBlockIdx()) continue;

        auto L0C_buf = loop_ping_flag ? L0C_base[L0C_PINGPONG_BUFFER_LEN] : L0C_base;
        auto LOOP_EVENT_ID = loop_ping_flag ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int64_t m_idx, n_idx;

        constexpr int64_t N_COL = 4;
        int64_t tile_block_loop = (n_loop + N_COL - 1) / N_COL;
        int64_t tile_block_idx = in_batch_idx / (N_COL * m_loop);
        int64_t in_tile_block_idx = in_batch_idx % (N_COL * m_loop);
        int64_t n_col = N_COL;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - N_COL * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * N_COL + in_tile_block_idx % n_col;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * ldc * N + m_idx * M0 + n_idx * N0 * ldc;
        int64_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int64_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int64_t m_round = ROUND(m_actual, 16);
        int64_t n_round = ROUND(n_actual, 16);

        int64_t mn_max = m_round > n_round ? m_round : n_round;
        int64_t L0AB_K0 = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        for (int64_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (transA != ASCBLAS_OP_N) {
                offset_a = batch_idx * M * lda + k_idx * K0 * M0 + m_idx * M0 * lda;
            } else {
                offset_a = batch_idx * lda * K + m_idx * M0 * K0 + k_idx * K0 * lda;
            }

            if (transB != ASCBLAS_OP_N) {
                offset_b = batch_idx * K * ldb + n_idx * N0 * K0 + k_idx * K0 * ldb;
            } else {
                offset_b = batch_idx * ldb * N + k_idx * K0 * N0 + n_idx * N0 * ldb;
            }

            int64_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int64_t k_round = ROUND(k_actual, 16);
            int64_t L0AB_k_loop = (k_actual + L0AB_K0 - 1) / L0AB_K0;

            auto L1_buf_a = k_loop_ping_flag ? L1_base_a : L1_base_a[L1_PINGPONG_BUFFER_LEN];
            auto L1_buf_b = k_loop_ping_flag ? L1_base_b : L1_base_b[L1_PINGPONG_BUFFER_LEN];
            auto K_LOOP_EVENT_ID = k_loop_ping_flag ? EVENT_ID0 : EVENT_ID1;

            WAIT_FLAG(MTE1, MTE2, K_LOOP_EVENT_ID);
            if (transA != ASCBLAS_OP_N) {
                ascblas_matrix_gm2cbuf_ND2nZ(L1_buf_a, gm_A[offset_a], K0, M0, k_actual, m_actual, K0);
            } else {
                ascblas_matrix_gm2cbuf_ND2nN(L1_buf_a, gm_A[offset_a], M0, K0, m_actual, k_actual, M0);
            }
            SET_FLAG(MTE2, MTE1, K_LOOP_EVENT_ID);

            WAIT_FLAG(MTE1, MTE2, K_LOOP_EVENT_ID + 2);
            if (transB != ASCBLAS_OP_N) {
                ascblas_matrix_gm2cbuf_ND2nN(L1_buf_b, gm_B[offset_b], N0, K0, n_actual, k_actual, N0);
            } else {
                ascblas_matrix_gm2cbuf_ND2nZ(L1_buf_b, gm_B[offset_b], K0, N0, k_actual, n_actual, K0);
            }
            SET_FLAG(MTE2, MTE1, K_LOOP_EVENT_ID + 2);

            for (int L0AB_k_idx = 0; L0AB_k_idx < L0AB_k_loop; L0AB_k_idx++) {
                int64_t L0AB_k_round = (L0AB_k_idx < L0AB_k_loop - 1) ? L0AB_K0 : k_round - L0AB_k_idx * L0AB_K0;
                int64_t L0AB_k_actual = (L0AB_k_idx < L0AB_k_loop - 1) ? L0AB_K0 : k_actual - L0AB_k_idx * L0AB_K0;

                auto mte1_mad_ping_flag = 1 - L0AB_k_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto L0A_buf = L0A_base[(L0AB_k_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                auto L0B_buf = L0B_base[(L0AB_k_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                if (L0AB_k_idx == 0) WAIT_FLAG(MTE2, MTE1, K_LOOP_EVENT_ID);
                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                if (transA != ASCBLAS_OP_N) {
                    auto L1_src_a = L1_buf_a[L0AB_k_idx * L0AB_K0 * M0];
                    for (int i = 0; i < L0AB_k_round / CUBE_K0; i++) {
                        AscendC::LoadData(L0B_buf[i * m_round * CUBE_K0], L1_src_a[i * CUBE_K0 * M0],
                            AscendC::LoadData2dParams(0, m_round / CUBE_M0, 1, 0, 0, false, inc));
                    }
                } else {
                    auto L1_src_a = L1_buf_a[L0AB_k_idx * L0AB_K0 * M0];
                    for (int i = 0; i < m_round / CUBE_M0; i++) {
                        AscendC::LoadDataWithTranspose(L0B_buf[i * CUBE_MATRIX_SIZE], L1_src_a[i * 2 * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(0, L0AB_k_round / (2 * CUBE_K0), M0 / CUBE_M0,
                                2 * m_round / CUBE_M0 - 1, m_round / CUBE_M0 - 1, inc));
                    }
                }
                if (L0AB_k_idx == L0AB_k_loop - 1) SET_FLAG(MTE1, MTE2, K_LOOP_EVENT_ID);

                if (L0AB_k_idx == 0) WAIT_FLAG(MTE2, MTE1, K_LOOP_EVENT_ID + 2);
                if (transB != ASCBLAS_OP_N) {
                    auto L1_src_b = L1_buf_b[L0AB_k_idx * L0AB_K0 * N0];
                    for (int i = 0; i < n_round / CUBE_N0; i++) {
                        AscendC::LoadDataWithTranspose(L0A_buf[i * L0AB_k_round * CUBE_N0], L1_src_b[i * 2 * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(0, L0AB_k_round / (2 * CUBE_K0), N0 / CUBE_N0, 1, 0, inc));
                    }
                } else {
                    auto L1_src_b = L1_buf_b[L0AB_k_idx * L0AB_K0 * N0];
                    for (int i = 0; i < n_round / CUBE_N0; i++) {
                        AscendC::LoadData(L0A_buf[i * L0AB_k_round * CUBE_N0], L1_src_b[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(0, L0AB_k_round / CUBE_K0, N0 / CUBE_N0, 0, 0, false, inc));
                    }
                }
                if (L0AB_k_idx == L0AB_k_loop - 1) SET_FLAG(MTE1, MTE2, K_LOOP_EVENT_ID + 2);

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && L0AB_k_idx == 0);
                if (init_c) WAIT_FLAG(FIX, M, LOOP_EVENT_ID);
                mad((__cc__ float *)L0C_buf.GetPhyAddr(), (__ca__ float *)L0A_buf.GetPhyAddr(),
                    (__cb__ float *)L0B_buf.GetPhyAddr(), n_round, L0AB_k_actual, m_round, 0, 1, 0, init_c);

                PIPE_BARRIER(M);
                SET_FLAG(M, MTE1, mte1_mad_event_id);
            }
            k_loop_ping_flag = 1 - k_loop_ping_flag;
        }

        SET_FLAG(M, FIX, LOOP_EVENT_ID);
        WAIT_FLAG(M, FIX, LOOP_EVENT_ID);

        if (is_AIV_AIC_PIPE) {
            WaitFlagDev(loop_ping_flag + 2);
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
                AIV_AIC_workspace[(AscendC::GetBlockIdx() * 2 + loop_ping_flag) * M0 * N0],
                L0C_buf, n_actual, m_actual, n_round, M0);
            FftsCrossCoreSync<PIPE_FIX, 2>(loop_ping_flag);
        } else {
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
                gm_C[offset_c], L0C_buf, n_actual, m_actual, n_round, ldc);
        }

        loop_ping_flag = 1 - loop_ping_flag;
        SET_FLAG(FIX, M, LOOP_EVENT_ID);
    }
    if (is_AIV_AIC_PIPE) {
        WaitFlagDev(0 + 2);
        WaitFlagDev(1 + 2);
    }

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);
}
#endif

#ifdef __DAV_C220_VEC__
__aicore__ __inline__ void splitMatrix(AscendC::GlobalTensor<float> dst_r, AscendC::GlobalTensor<float> dst_i,
                                        AscendC::GlobalTensor<float> src, int64_t M, int64_t N, int64_t dst_pad, int64_t src_pad)
{
    int64_t M0 = 128, N0 = 128;
    int64_t NUM_PER_REPEAT = 6 * 1024;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> buf0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> buf1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);
    int64_t v_block_num = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    int64_t v_id = AscendC::GetBlockIdx();

    int64_t m_repeats = (M + NUM_PER_REPEAT - 1) / NUM_PER_REPEAT;
    if (m_repeats == 0) m_repeats = 1;
    int64_t M_round = ROUND(M, NUM_ELE_PERBLOCK);
    if (M_round == 0) M_round = 1;

    int64_t n_per_repeat = NUM_PER_REPEAT / M_round + int(NUM_PER_REPEAT < M_round);
    int64_t n_repeats = (N + n_per_repeat - 1) / n_per_repeat;
    int64_t repeats = n_repeats * m_repeats;
    int64_t repeats_per_core = repeats / v_block_num + int(v_id < repeats % v_block_num);

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);
    SET_FLAG(V, MTE2, EVENT_ID2);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(MTE3, V, EVENT_ID3);

    for (int64_t pi = 0; pi < repeats_per_core; pi++) {
        int64_t i = pi * v_block_num + v_id;
        int64_t c_id = i / m_repeats;
        int64_t r_id = i % m_repeats;
        int64_t m_len = NUM_PER_REPEAT;
        if (r_id + 1 == m_repeats) m_len = M - r_id * NUM_PER_REPEAT;
        int64_t m_round = ROUND(m_len, NUM_ELE_PERBLOCK);
        int64_t n_len = n_per_repeat;
        if (c_id + 1 == n_repeats) n_len = N - c_id * n_per_repeat;

        AscendC::LocalTensor<float> complex_buf = (pi & 1) ? buf1 : buf0;
        AscendC::LocalTensor<float> real_buf = complex_buf[2 * NUM_PER_REPEAT];
        AscendC::LocalTensor<float> imag_buf = real_buf[NUM_PER_REPEAT];
        auto event_id = (pi & 1) ? 2 : 0;

        WAIT_FLAG(V, MTE2, event_id);
        ascblas_matrix_gm2ubuf(complex_buf, src[(c_id * n_per_repeat * src_pad + r_id * NUM_PER_REPEAT) * 2],
                               m_len * 2, n_len, src_pad * 2, m_round * 2);
        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE3, V, event_id);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(real_buf.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(complex_buf.GetPhyAddr()), nullptr,
                  (m_round * 2 * n_len + 63) / 64, 1, 1, 8, 8);
        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(MTE3, V, event_id + 1);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(imag_buf.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(complex_buf.GetPhyAddr()), nullptr,
                  (m_round * 2 * n_len + 63) / 64, 1, 2, 8, 8);
        SET_FLAG(V, MTE3, event_id + 1);
        SET_FLAG(V, MTE2, event_id);
        WAIT_FLAG(V, MTE3, event_id);
        for (int ni = 0; ni < n_len; ++ni) {
            int n_ind = c_id * n_per_repeat + ni;
            ub_to_gm<ArchType::ASCEND_V220, float>(
                dst_r[n_ind / N0 * N0 * dst_pad + n_ind % N0 * M0 + r_id * NUM_PER_REPEAT * N0],
                real_buf[ni * m_round], 0, (m_round + M0 - 1) / M0, M0 / NUM_ELE_PERBLOCK, 0, (M0 * N0 - M0) / NUM_ELE_PERBLOCK);
        }
        SET_FLAG(MTE3, V, event_id);
        WAIT_FLAG(V, MTE3, event_id + 1);
        for (int ni = 0; ni < n_len; ++ni) {
            int n_ind = c_id * n_per_repeat + ni;
            ub_to_gm<ArchType::ASCEND_V220, float>(
                dst_i[n_ind / N0 * N0 * dst_pad + n_ind % N0 * M0 + r_id * NUM_PER_REPEAT * N0],
                imag_buf[ni * m_round], 0, (m_round + M0 - 1) / M0, M0 / NUM_ELE_PERBLOCK, 0, (M0 * N0 - M0) / NUM_ELE_PERBLOCK);
        }
        SET_FLAG(MTE3, V, event_id + 1);
    }
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
    WAIT_FLAG(V, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(MTE3, V, EVENT_ID3);

}

__aicore__ __inline__ void ascblasCgemmPre(ascblasOperation_t transA, ascblasOperation_t transB, int64_t M, int64_t N,
                                           int64_t K, ascComplex alpha, AscendC::GlobalTensor<float> d_A, int64_t lda,
                                           AscendC::GlobalTensor<float> d_B, int64_t ldb, ascComplex beta,
                                           AscendC::GlobalTensor<float> d_C, int64_t ldc, int64_t lda_pad, int64_t ldb_pad,
                                           AscendC::GlobalTensor<float> d_A_r, AscendC::GlobalTensor<float> d_A_i,
                                           AscendC::GlobalTensor<float> d_B_r, AscendC::GlobalTensor<float> d_B_i,
                                           AscendC::GlobalTensor<float> d_C_rr, AscendC::GlobalTensor<float> d_C_ri,
                                           AscendC::GlobalTensor<float> d_C_ir, AscendC::GlobalTensor<float> d_C_ii)
{
    if (transA == ASCBLAS_OP_N) splitMatrix(d_A_r, d_A_i, d_A, M, K, lda_pad, lda);
    else splitMatrix(d_A_r, d_A_i, d_A, K, M, lda_pad, lda);
    if (transB == ASCBLAS_OP_N) splitMatrix(d_B_r, d_B_i, d_B, K, N, ldb_pad, ldb);
    else splitMatrix(d_B_r, d_B_i, d_B, N, K, ldb_pad, ldb);
}

__aicore__ __inline__ void mulSComplex(AscendC::LocalTensor<float> src_r, AscendC::LocalTensor<float> src_i,
                                        AscendC::LocalTensor<float> tmp_r, AscendC::LocalTensor<float> tmp_i,
                                        int64_t n_blocks, ascComplex beta)
{
    muls_v<ArchType::ASCEND_V220, float>(tmp_r, src_r, beta.real, (n_blocks + 7) / 8, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(tmp_i, src_i, beta.imag, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    sub_v<ArchType::ASCEND_V220, float>(tmp_r, tmp_r, tmp_i, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(tmp_i, src_r, beta.imag, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    ub_to_ub<ArchType::ASCEND_V220, float>(src_r, tmp_r, 0, 1, n_blocks, 0, 0);
    PIPE_BARRIER(V);
    muls_v<ArchType::ASCEND_V220, float>(tmp_r, src_i, beta.real, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    add_v<ArchType::ASCEND_V220, float>(tmp_i, tmp_i, tmp_r, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    ub_to_ub<ArchType::ASCEND_V220, float>(src_i, tmp_i, 0, 1, n_blocks, 0, 0);
}

__aicore__ __inline__ void ascblasCgemmFinal(ascblasOperation_t transA, ascblasOperation_t transB, int64_t M, int64_t N,
                                             int64_t K, ascComplex alpha, AscendC::GlobalTensor<float> d_A, int64_t lda,
                                             AscendC::GlobalTensor<float> d_B, int64_t ldb, ascComplex beta,
                                             AscendC::GlobalTensor<float> d_C, int64_t ldc, int64_t lda_pad, int64_t ldb_pad,
                                             AscendC::GlobalTensor<float> d_A_r, AscendC::GlobalTensor<float> d_A_i,
                                             AscendC::GlobalTensor<float> d_B_r, AscendC::GlobalTensor<float> d_B_i,
                                             AscendC::GlobalTensor<float> d_C_rr, AscendC::GlobalTensor<float> d_C_ri,
                                             AscendC::GlobalTensor<float> d_C_ir, AscendC::GlobalTensor<float> d_C_ii)
{
    constexpr int64_t NUM_PER_REPEAT = 6 * 1024;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> complex_buf = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> complex_buf_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> complex_buf_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(NUM_PER_REPEAT * 1 * sizeof(float));
    AscendC::LocalTensor<float> real_buf = buf.GetBuffer<BufferType::ASCEND_UB, float>(NUM_PER_REPEAT * 2 * sizeof(float));
    AscendC::LocalTensor<float> imag_buf = buf.GetBuffer<BufferType::ASCEND_UB, float>(NUM_PER_REPEAT * 3 * sizeof(float));
    AscendC::LocalTensor<uint32_t> mask_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(NUM_PER_REPEAT * 4 * sizeof(float));
    AscendC::LocalTensor<float> real_tbuf = buf.GetBuffer<BufferType::ASCEND_UB, float>(NUM_PER_REPEAT * 6 * sizeof(float));
    AscendC::LocalTensor<float> imag_tbuf = buf.GetBuffer<BufferType::ASCEND_UB, float>(NUM_PER_REPEAT * 7 * sizeof(float));

    int v_block_num = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    if (v_block_num == 0) v_block_num = 1;
    int v_id = AscendC::GetBlockIdx();

    int64_t m_repeats = (M + NUM_PER_REPEAT - 1) / NUM_PER_REPEAT;
    if (m_repeats == 0) m_repeats = 1;
    int64_t M_round = ROUND(M, NUM_ELE_PERBLOCK);
    if (M_round == 0) M_round = 1;
    int64_t n_per_repeat = NUM_PER_REPEAT / M_round + int(NUM_PER_REPEAT < M_round);
    int64_t n_repeats = (N + n_per_repeat - 1) / n_per_repeat;
    int64_t repeats = n_repeats * m_repeats;
    int64_t repeats_per_core = repeats / v_block_num + int(v_id < repeats % v_block_num);

    auto real_base = (uintptr_t)(NUM_PER_REPEAT * 2 * sizeof(float));
    auto img_base = (uintptr_t)(NUM_PER_REPEAT * 3 * sizeof(float));
    int k = 0;
    for (int64_t j = 0; j < NUM_PER_REPEAT; j++) {
        mask_buf.SetValue(k++, real_base + j * sizeof(float));
        mask_buf.SetValue(k++, img_base + j * sizeof(float));
    }

    PIPE_BARRIER(ALL);
    SET_FLAG(MTE3, MTE2, EVENT_ID0);

    for (int64_t pi = 0; pi < repeats_per_core; pi++) {
        int64_t i = pi * v_block_num + v_id;
        int64_t c_id = i / m_repeats;
        int64_t r_id = i % m_repeats;
        int64_t m_len = NUM_PER_REPEAT;
        if (r_id + 1 == m_repeats) m_len = M - r_id * NUM_PER_REPEAT;
        int64_t m_round = ROUND(m_len, NUM_ELE_PERBLOCK);
        int64_t n_len = n_per_repeat;
        if (c_id + 1 == n_repeats) n_len = N - c_id * n_per_repeat;
        int64_t n_blocks = (m_round * n_len + NUM_ELE_PERBLOCK - 1) / NUM_ELE_PERBLOCK;

        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        ascblas_matrix_gm2ubuf(complex_buf, d_C[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT) * 2],
                               m_len * 2, n_len, ldc * 2, m_round * 2);
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(real_buf.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(complex_buf.GetPhyAddr()), nullptr,
                  (m_round * 2 * n_len + 63) / 64, 1, 1, 8, 8);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(imag_buf.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(complex_buf.GetPhyAddr()), nullptr,
                  (m_round * 2 * n_len + 63) / 64, 1, 2, 8, 8);

        PIPE_BARRIER(V);
        mulSComplex(real_buf, imag_buf, real_tbuf, imag_tbuf, n_blocks, beta);

        SET_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        ascblas_matrix_gm2ubuf(complex_buf_real, d_C_rr[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT)], m_len, n_len, ldc, m_round);
        ascblas_matrix_gm2ubuf(complex_buf_imag, d_C_ri[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT)], m_len, n_len, ldc, m_round);
        ascblas_matrix_gm2ubuf(imag_tbuf, d_C_ir[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT)], m_len, n_len, ldc, m_round);
        ascblas_matrix_gm2ubuf(real_tbuf, d_C_ii[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT)], m_len, n_len, ldc, m_round);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        if ((transA == ASCBLAS_OP_C) != (transB == ASCBLAS_OP_C))
            add_v<ArchType::ASCEND_V220, float>(complex_buf_real, complex_buf_real, real_tbuf, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
        else
            sub_v<ArchType::ASCEND_V220, float>(complex_buf_real, complex_buf_real, real_tbuf, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
        if (transB == ASCBLAS_OP_C)
            muls_v<ArchType::ASCEND_V220, float>(complex_buf_imag, complex_buf_imag, -1.0f, (n_blocks + 7) / 8, 1, 1, 8, 8);
        PIPE_BARRIER(V);

        if (transA == ASCBLAS_OP_C)
            sub_v<ArchType::ASCEND_V220, float>(complex_buf_imag, complex_buf_imag, imag_tbuf, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
        else
            add_v<ArchType::ASCEND_V220, float>(complex_buf_imag, complex_buf_imag, imag_tbuf, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        mulSComplex(complex_buf_real, complex_buf_imag, real_tbuf, imag_tbuf, n_blocks, alpha);

        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V220, float>(real_buf, real_buf, complex_buf_real, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
        add_v<ArchType::ASCEND_V220, float>(imag_buf, imag_buf, complex_buf_imag, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        AscendC::Gather(complex_buf, complex_buf, mask_buf, (uint32_t)0, (m_round * 2 * n_len + 63) / 64 * 64);

        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ascblas_matrix_ubuf2gm(d_C[(c_id * n_per_repeat * ldc + r_id * NUM_PER_REPEAT) * 2], complex_buf,
                               m_len * 2, n_len, m_round * 2, ldc * 2);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
}
#endif

extern "C" __global__ __aicore__ void cgemm(__gm__ uint8_t *__restrict__ d_A, __gm__ uint8_t *__restrict__ d_B,
                                            __gm__ uint8_t *__restrict__ d_A_r, __gm__ uint8_t *__restrict__ d_A_i,
                                            __gm__ uint8_t *__restrict__ d_B_r, __gm__ uint8_t *__restrict__ d_B_i,
                                            __gm__ uint8_t *__restrict__ d_C_rr, __gm__ uint8_t *__restrict__ d_C_ri,
                                            __gm__ uint8_t *__restrict__ d_C_ir, __gm__ uint8_t *__restrict__ d_C_ii,
                                            __gm__ uint8_t *__restrict__ d_C, __gm__ uint8_t *__restrict__ workspace,
                                            __gm__ uint8_t *__restrict__ tilingGm)
{
#ifdef __DAV_C220_CUBE__
    
    SetPadding(0);
    AscendC::SetAtomicNone();
    AscendC::SetNdParaImpl(0x1);
    
    auto tiling_buf = tilingGm;
    int64_t M = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    int64_t N = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8));
    int64_t K = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    int64_t transA = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 24));
    int64_t transB = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 32));
    int64_t lda = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 40));
    int64_t ldb = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 48));
    int64_t ldc = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 56));
    int64_t lda_pad = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 64));
    int64_t ldb_pad = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 72));

    PIPE_BARRIER(ALL);

    AscendC::GlobalTensor<float> d_A_r_tensor, d_A_i_tensor, d_B_r_tensor, d_B_i_tensor;
    AscendC::GlobalTensor<float> d_C_rr_tensor, d_C_ri_tensor, d_C_ir_tensor, d_C_ii_tensor;
    AscendC::GlobalTensor<float> workspace_tensor;
    d_A_r_tensor.SetGlobalBuffer((__gm__ float *)d_A_r);
    d_A_i_tensor.SetGlobalBuffer((__gm__ float *)d_A_i);
    d_B_r_tensor.SetGlobalBuffer((__gm__ float *)d_B_r);
    d_B_i_tensor.SetGlobalBuffer((__gm__ float *)d_B_i);
    d_C_rr_tensor.SetGlobalBuffer((__gm__ float *)d_C_rr);
    d_C_ri_tensor.SetGlobalBuffer((__gm__ float *)d_C_ri);
    d_C_ir_tensor.SetGlobalBuffer((__gm__ float *)d_C_ir);
    d_C_ii_tensor.SetGlobalBuffer((__gm__ float *)d_C_ii);
    workspace_tensor.SetGlobalBuffer((__gm__ float *)workspace);

    ascblasOperation_t _transA = transA == 0 ? ASCBLAS_OP_N : (transA == 1 ? ASCBLAS_OP_T : ASCBLAS_OP_C);
    ascblasOperation_t g_transB = transB == 0 ? ASCBLAS_OP_N : (transB == 1 ? ASCBLAS_OP_T : ASCBLAS_OP_C);

    WaitFlagDev(0);
    ascblasSmatmul(_transA, g_transB, M, N, K, d_A_r_tensor, lda_pad, d_B_r_tensor, ldb_pad, d_C_rr_tensor, ldc, 1, 128, 128, 128, 0, workspace_tensor);
    ascblasSmatmul(_transA, g_transB, M, N, K, d_A_r_tensor, lda_pad, d_B_i_tensor, ldb_pad, d_C_ri_tensor, ldc, 1, 128, 128, 128, 0, workspace_tensor);
    ascblasSmatmul(_transA, g_transB, M, N, K, d_A_i_tensor, lda_pad, d_B_r_tensor, ldb_pad, d_C_ir_tensor, ldc, 1, 128, 128, 128, 0, workspace_tensor);
    ascblasSmatmul(_transA, g_transB, M, N, K, d_A_i_tensor, lda_pad, d_B_i_tensor, ldb_pad, d_C_ii_tensor, ldc, 1, 128, 128, 128, 0, workspace_tensor);

    FftsCrossCoreSync<PIPE_FIX, 0>(1);
    WaitFlagDev(1);
    FftsCrossCoreSync<PIPE_FIX, 2>(1);

    PIPE_BARRIER(ALL);
#endif

#ifdef __DAV_C220_VEC__
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    auto tiling_buf = tilingGm;
    int64_t M = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    int64_t N = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8));
    int64_t K = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    int64_t transA = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 24));
    int64_t transB = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 32));
    int64_t lda = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 40));
    int64_t ldb = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 48));
    int64_t ldc = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 56));
    int64_t lda_pad = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 64));
    int64_t ldb_pad = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 72));
    float alpha_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 80));
    float alpha_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 84));
    float beta_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 88));
    float beta_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 92));

    PIPE_BARRIER(ALL);
    ascComplex alpha{alpha_r, alpha_i};
    ascComplex beta{beta_r, beta_i};

    ascblasOperation_t _transA = transA == 0 ? ASCBLAS_OP_N : (transA == 1 ? ASCBLAS_OP_T : ASCBLAS_OP_C);
    ascblasOperation_t g_transB = transB == 0 ? ASCBLAS_OP_N : (transB == 1 ? ASCBLAS_OP_T : ASCBLAS_OP_C);

    AscendC::GlobalTensor<float> d_A_tensor, d_B_tensor, d_C_tensor;
    AscendC::GlobalTensor<float> d_A_r_tensor, d_A_i_tensor, d_B_r_tensor, d_B_i_tensor;
    AscendC::GlobalTensor<float> d_C_rr_tensor, d_C_ri_tensor, d_C_ir_tensor, d_C_ii_tensor;
    d_A_tensor.SetGlobalBuffer((__gm__ float *)d_A);
    d_B_tensor.SetGlobalBuffer((__gm__ float *)d_B);
    d_C_tensor.SetGlobalBuffer((__gm__ float *)d_C);
    d_A_r_tensor.SetGlobalBuffer((__gm__ float *)d_A_r);
    d_A_i_tensor.SetGlobalBuffer((__gm__ float *)d_A_i);
    d_B_r_tensor.SetGlobalBuffer((__gm__ float *)d_B_r);
    d_B_i_tensor.SetGlobalBuffer((__gm__ float *)d_B_i);
    d_C_rr_tensor.SetGlobalBuffer((__gm__ float *)d_C_rr);
    d_C_ri_tensor.SetGlobalBuffer((__gm__ float *)d_C_ri);
    d_C_ir_tensor.SetGlobalBuffer((__gm__ float *)d_C_ir);
    d_C_ii_tensor.SetGlobalBuffer((__gm__ float *)d_C_ii);

    ascblasCgemmPre(_transA, g_transB, M, N, K, alpha, d_A_tensor, lda, d_B_tensor, ldb, beta, d_C_tensor, ldc,
                    lda_pad, ldb_pad, d_A_r_tensor, d_A_i_tensor, d_B_r_tensor, d_B_i_tensor,
                    d_C_rr_tensor, d_C_ri_tensor, d_C_ir_tensor, d_C_ii_tensor);
    FftsCrossCoreSync<PIPE_MTE3, 0>(0);
    WaitFlagDev(0);
    FftsCrossCoreSync<PIPE_MTE3, 2>(0);
    WaitFlagDev(1);
    ascblasCgemmFinal(_transA, g_transB, M, N, K, alpha, d_A_tensor, lda, d_B_tensor, ldb, beta, d_C_tensor, ldc,
                      lda_pad, ldb_pad, d_A_r_tensor, d_A_i_tensor, d_B_r_tensor, d_B_i_tensor,
                      d_C_rr_tensor, d_C_ri_tensor, d_C_ir_tensor, d_C_ii_tensor);
    PIPE_BARRIER(ALL);
#endif
}

void cgemm_kernel_do(GM_ADDR d_A, GM_ADDR d_B, GM_ADDR d_A_r, GM_ADDR d_A_i,
                     GM_ADDR d_B_r, GM_ADDR d_B_i, GM_ADDR d_C_rr, GM_ADDR d_C_ri,
                     GM_ADDR d_C_ir, GM_ADDR d_C_ii, GM_ADDR d_C, GM_ADDR workspace,
                     GM_ADDR tilingGm, uint32_t numBlocks, void *stream)
{
    cgemm<<<numBlocks, nullptr, stream>>>(d_A, d_B, d_A_r, d_A_i, d_B_r, d_B_i,
                                          d_C_rr, d_C_ri, d_C_ir, d_C_ii, d_C, workspace, tilingGm);
}