/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(float);
constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(float);
constexpr int64_t CUBE_K0 = 32 / sizeof(float);
constexpr int64_t CUBE_M0 = 16;
constexpr int64_t CUBE_N0 = 16;
constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;

#if __DAV_C220_CUBE__

__aicore__ __inline__ void load_matrix_zN(__cbuf__ float *dst, __gm__ float *src, int32_t R, int32_t C,
                                          int32_t valid_row, int32_t valid_col, size_t stride)
{
    constexpr int C0 = 32 / sizeof(float);
    constexpr int STRIDE_LIMIT = 65536;

    if (stride < STRIDE_LIMIT) {
        copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src,
                                         static_cast<uint8_t>(0),
                                         static_cast<uint16_t>(1),
                                         static_cast<uint16_t>(valid_row),
                                         static_cast<uint16_t>(valid_col),
                                         static_cast<uint16_t>(0),
                                         static_cast<uint16_t>(stride),
                                         static_cast<uint16_t>(R),
                                         static_cast<uint16_t>(1),
                                         static_cast<uint16_t>(0)
        );
    } else {
        for (int i = 0; i < valid_row; i++) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst + i * C0, src + i * stride,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(R),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(0)
            );
        }
    }
}

__aicore__ __inline__ void load_matrix_zZ(__cbuf__ float *dst, __gm__ float *src, int32_t R, int32_t C,
                                          int32_t valid_row, int32_t valid_col, size_t stride)
{
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(float);
    constexpr int STRIDE_LIMIT = 65536;

    int64_t srcNdStride = R0 * stride;
    int64_t srcNStride = stride;
    if (srcNdStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        if (ndNum > 0) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(ndNum),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(R0 * stride),
                                             static_cast<uint16_t>(srcNStride),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(R0 * C)
            );
        }
        if (remains > 0) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst + ndNum * R0 * C, src + ndNum * R0 * stride,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(remains),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(srcNStride),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(0)
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst + i * R0 * C, src + i * R0 * stride,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(srcNStride),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(0)
            );
        }
        if (remains > 0) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst + ndNum * R0 * C, src + ndNum * R0 * stride,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(remains),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(srcNStride),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(0)
            );
        }
    } else {
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;
            int32_t idxInR0 = i % R0;
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst + idxR0 * R0 * C + idxInR0 * C0, src + i * stride,
                                             static_cast<uint8_t>(0),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(1),
                                             static_cast<uint16_t>(valid_col),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(R0),
                                             static_cast<uint16_t>(0),
                                             static_cast<uint16_t>(0)
            );
        }
    }
}

__aicore__ __inline__ int64_t GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    int64_t modeOffset = 4;
    int64_t flagOffset = 8;
    return 1 | (mode << modeOffset) | (flagId << flagOffset);
}

__aicore__ __inline__ void strmm_mix_aic(__gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
                                         __gm__ float *__restrict__ gm_c, uint32_t M, uint32_t N, uint32_t K,
                                         uint32_t trans_a, uint32_t trans_b, uint32_t lessFlag,
                                         __gm__ float *__restrict__ AIV_AIC_workspace)
{
    set_padding(0);
    set_atomic_none();
    uint64_t config = 0x1;
    set_nd_para(config);

    auto l1_base_a = reinterpret_cast<__cbuf__ float *>((uintptr_t)0);
    auto l1_base_b = reinterpret_cast<__cbuf__ float *>((uintptr_t)(128 * 1024));

    auto l0a_base = reinterpret_cast<__ca__ float *>((uintptr_t)0);
    auto l0b_base = reinterpret_cast<__cb__ float *>((uintptr_t)0);
    auto l0c_buf = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    int32_t batchSize, M0, N0, K0;

    batchSize = 1;

    M0 = 128;
    N0 = 128;
    K0 = 128;

    int64_t lda = M;
    int64_t ldb = K;
    int64_t ldc = M;

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;
    int32_t loop_ping_flag = 1;

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    int32_t blockNum = get_block_num();
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % blockNum != get_block_idx()) {
            continue;
        }

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int64_t m_idx;
        int64_t n_idx;

        m_idx = loop_idx / n_loop;
        n_idx = loop_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * (int64_t)M * (int64_t)N + m_idx * (int64_t)M0 * (int64_t)N + n_idx * (int64_t)N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - (int32_t)m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - (int32_t)n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                offset_a = batch_idx * M * K + k_idx * K0 * M + m_idx * M0;
            } else {
                offset_a = batch_idx * M * K + m_idx * M0 * K + k_idx * K0;
            }

            if (trans_b) {
                offset_b = batch_idx * K * N + n_idx * N0 * K + k_idx * K0;
            } else {
                offset_b = batch_idx * K * N + k_idx * K0 * N + n_idx * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = ping_flag ? l1_base_a : l1_base_a + L1_PINGPONG_BUFFER_LEN;
            auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            wait_flag(PIPE_MTE1, PIPE_MTE2, event_id);
            if (trans_a) {
                if (M == 1) {
                    copy_gm_to_cbuf(l1_buf_a, gm_a + offset_a,
                                    0,
                                    1,
                                    (k_actual + C0_SIZE - 1) / C0_SIZE,
                                    0,
                                    0,
                                    PAD_NONE
                    );
                } else {
                    load_matrix_zZ(l1_buf_a, gm_a + offset_a, K0, M0, k_actual, m_actual, M);
                }
            } else {
                if (m_actual == 1) {
                    copy_gm_to_cbuf(l1_buf_a, gm_a + offset_a,
                                    0,
                                    1,
                                    (k_actual + C0_SIZE - 1) / C0_SIZE,
                                    0,
                                    0,
                                    PAD_NONE
                    );
                } else {
                    load_matrix_zZ(l1_buf_a, gm_a + offset_a, M0, K0, m_actual, k_actual, K);
                }
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, event_id);

            wait_flag(PIPE_MTE1, PIPE_MTE2, event_id + 2);
            if (trans_b) {
                load_matrix_zZ(l1_buf_b, gm_b + offset_b, N0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ(l1_buf_b, gm_b + offset_b, K0, N0, k_actual, n_actual, N);
            }

            set_flag(PIPE_MTE2, PIPE_MTE1, event_id + 2);

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_buf = l0a_base + (k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;
                auto l0b_buf = l0b_base + (k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;

                if (k_part_idx == 0) {
                    wait_flag(PIPE_MTE2, PIPE_MTE1, event_id);
                }
                wait_flag(PIPE_M, PIPE_MTE1, mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    load_cbuf_to_ca(l0a_buf, l1_buf_a + k_part_idx * k_part_len,
                                    0,
                                    (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,
                                    1,
                                    0,
                                    0,
                                    false,
                                    inc
                    );

                } else {
                    if (trans_a) {
                        auto l1_src_a = l1_buf_a + k_part_idx * k_part_len * M0;
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            load_cbuf_to_ca_transpose(l0a_buf + i * k0_round * BLOCK_SIZE,
                                                      l1_src_a + i * 2 * CUBE_MATRIX_SIZE,
                                                      0,
                                                      k0_round / BLOCK_SIZE,
                                                      M0 / BLOCK_SIZE,
                                                      1,
                                                      inc,
                                                      0
                            );
                        }
                    } else {
                        auto l1_src_a = l1_buf_a + k_part_idx * k_part_len * BLOCK_SIZE;
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            load_cbuf_to_ca(l0a_buf + i * k0_round * BLOCK_SIZE, l1_src_a + i * BLOCK_SIZE * K0,
                                            0,
                                            k0_round / C0_SIZE,
                                            1,
                                            0,
                                            0,
                                            false,
                                            inc
                            );
                        }
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, event_id);
                }

                if (k_part_idx == 0) {
                    wait_flag(PIPE_MTE2, PIPE_MTE1, event_id + 2);
                }
                if (trans_b) {
                    auto l1_src_b = l1_buf_b + k_part_idx * k_part_len * BLOCK_SIZE;
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        load_cbuf_to_cb(l0b_buf + i * n_round * C0_SIZE, l1_src_b + i * CUBE_MATRIX_SIZE,
                                        0,
                                        n_round / BLOCK_SIZE,
                                        K0 / C0_SIZE,
                                        0,
                                        0,
                                        false,
                                        inc
                        );
                    }
                } else {
                    auto l1_src_b = l1_buf_b + k_part_idx * k_part_len * N0;
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        load_cbuf_to_cb_transpose(l0b_buf + i * CUBE_MATRIX_SIZE, l1_src_b + i * 2 * CUBE_MATRIX_SIZE,
                                                  0,
                                                  k0_round / BLOCK_SIZE,
                                                  N0 / BLOCK_SIZE,
                                                  2 * n_round / BLOCK_SIZE - 1,
                                                  inc,
                                                  n_round / BLOCK_SIZE - 1
                        );
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, event_id + 2);
                }

                set_flag(PIPE_MTE1, PIPE_M, mte1_mad_event_id);
                wait_flag(PIPE_MTE1, PIPE_M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    mad(l0c_buf, l0a_buf, l0b_buf,
                        16,
                        k0_actual,
                        n_actual,
                        0,
                        1,
                        0,
                        init_c
                    );
                } else {
                    mad(l0c_buf, l0a_buf, l0b_buf,
                        m_actual,
                        k0_actual,
                        n_actual,
                        0,
                        1,
                        0,
                        init_c
                    );
                }

                pipe_barrier(PIPE_M);
                set_flag(PIPE_M, PIPE_MTE1, mte1_mad_event_id);
            }

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            ping_flag = 1 - ping_flag;
        }

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        copy_matrix_cc_to_gm(gm_c + offset_c, l0c_buf,
                             0,
                             n_actual,
                             m_actual,
                             N,
                             m_round,
                             0,
                             NoQuant,
                             0,
                             false,
                             true
        );
        ffts_cross_core_sync(PIPE_FIX, GET_FFST_MSG(2, loop_ping_flag));

        loop_ping_flag = 1 - loop_ping_flag;

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    pipe_barrier(PIPE_ALL);
}

#endif

#if __DAV_C220_VEC__

__aicore__ __inline__ void ascblas_matrix_gm2ubuf_cce(__ubuf__ float *dst, __gm__ float *src, int64_t m_actual,
                                                      int64_t n_actual, size_t srcStride, size_t dstStride)
{
    for (int i = 0; i < m_actual; i++) {
        copy_gm_to_ubuf_align_b32(dst + i * dstStride, src + i * srcStride, 0, 1, n_actual * sizeof(float), 0, 0, 0, 0);
    }
}

__aicore__ __inline__ void ascblas_matrix_ubuf2gm_cce(__gm__ float *dst, __ubuf__ float *src,
                                                      int64_t m_actual, int64_t n_actual, size_t srcStride, size_t dstStride)
{
    for (int64_t i = 0; i < m_actual; i++) {
        copy_ubuf_to_gm_align_b32(dst + i * dstStride, src + i * srcStride, 0, 1, n_actual * sizeof(float), 0, 0, 0, 0);
    }
}

__aicore__ __inline__ void strmm_mix_aiv(__gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
                                         __gm__ float *__restrict__ gm_C, uint32_t M, uint32_t N, uint32_t K,
                                         float alpha, uint32_t lessFlag, __gm__ float *__restrict__ AIV_AIC_workspace)
{
    int64_t ldc = M;
    int64_t batchSize, trans_a, trans_b, M0, N0, K0;

    batchSize = 1;

    trans_a = 0;
    trans_b = 0;
    M0 = 128;
    N0 = 128;
    K0 = 128;

    auto ubufC1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)0);
    auto ubufC2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(32 * 1024));
    auto ubufAB1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(64 * 1024));
    auto ubufAB2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(96 * 1024));
    auto ubufD1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(128 * 1024));
    auto ubufD2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(160 * 1024));

    int64_t m_loop = (M + M0 - 1) / M0;
    int64_t n_loop = (N + N0 - 1) / N0;
    int64_t k_loop = (K + K0 - 1) / K0;
    int64_t loop = batchSize * m_loop * n_loop;

    int64_t loop_ping_flag = 1;
    int64_t k_loop_ping_flag = 1;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    int64_t blockNum = get_block_num();
    for (int64_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % blockNum != get_block_idx()) {
            continue;
        }

        auto ubufC = loop_ping_flag ? ubufC1 : ubufC2;
        auto C_EVENT_ID = loop_ping_flag ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int64_t m_idx;
        int64_t n_idx;

        m_idx = loop_idx / n_loop;
        n_idx = loop_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - (int32_t)m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - (int32_t)n_idx * N0) : N0;

        wait_flag(PIPE_MTE3, PIPE_MTE2, C_EVENT_ID);

        wait_flag_dev(loop_ping_flag);

        if (get_subblockid() == 0) {
            ascblas_matrix_gm2ubuf_cce(ubufC, gm_C + offset_c, M0 / 2 < m_actual ? M0 / 2 : m_actual, n_actual, N, N0);
        } else if (get_subblockid() == 1 && M0 / 2 < m_actual) {
            ascblas_matrix_gm2ubuf_cce(ubufC, gm_C + offset_c + get_subblockid() * N * M0 / 2, m_actual - M0 / 2,
                                       n_actual, N, N0);
        }

        set_flag(PIPE_MTE2, PIPE_V, C_EVENT_ID);
        wait_flag(PIPE_MTE2, PIPE_V, C_EVENT_ID);

        vmuls(ubufC, ubufC, alpha, M0 * N0 / 2 / (NUM_ELE_PERBLOCK * 8), 1, 1, 8, 8);

        set_flag(PIPE_V, PIPE_MTE3, C_EVENT_ID);
        wait_flag(PIPE_V, PIPE_MTE3, C_EVENT_ID);

        if (get_subblockid() == 0) {
            ascblas_matrix_ubuf2gm_cce(gm_C + offset_c, ubufC, M0 / 2 < m_actual ? M0 / 2 : m_actual, n_actual, N0, N);
        } else if (get_subblockid() == 1 && M0 / 2 < m_actual) {
            ascblas_matrix_ubuf2gm_cce(gm_C + offset_c + get_subblockid() * N * M0 / 2, ubufC, m_actual - M0 / 2,
                                       n_actual, N0, N);
        }

        set_flag(PIPE_MTE3, PIPE_MTE2, C_EVENT_ID);

        loop_ping_flag = 1 - loop_ping_flag;
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}
#endif

extern "C" __global__ __aicore__ void strmm(__gm__ float *__restrict__ d_A,
                                            __gm__ float *__restrict__ d_B, __gm__ float *__restrict__ d_C,
                                            __gm__ float *__restrict__ workspace,
                                            __gm__ uint32_t *__restrict__ tiling_gm)
{
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);

    uint32_t side = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    uint32_t uplo = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));
    uint32_t transa = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8));
    uint32_t transb = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12));
    uint32_t diag = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    uint32_t M = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 20));
    uint32_t N = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 24));
    uint32_t K = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 28));
    uint32_t lessFlag = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 32));
    float alpha = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 36));

#if __DAV_C220_CUBE__
    set_padding(0);
    set_atomic_none();
    set_nd_para(0x1);
    strmm_mix_aic(d_A, d_B, d_C, M, N, K, transa, transb, lessFlag, workspace);
#endif

#if __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    strmm_mix_aiv(d_A, d_B, d_C, M, N, K, alpha, lessFlag, workspace);
#endif
}

void strmm_kernel_do(GM_ADDR d_A, GM_ADDR d_B, GM_ADDR d_C, GM_ADDR workspace,
                     GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream)
{
    strmm<<<numBlocks, nullptr, stream>>>((__gm__ float *)d_A, (__gm__ float *)d_B,
                                          (__gm__ float *)d_C, (__gm__ float *)workspace,
                                          (__gm__ uint32_t *)tilingGm);
}