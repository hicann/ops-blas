/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use the License for the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_operator.h"
#include "../common/common.h"
#include "../common/iterator.h"
#include "../common/simd.h"

#define __inline__ __inline__ __attribute__((always_inline))

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_uint32(
    AscendC::LocalTensor<uint32_t> dst,
    AscendC::GlobalTensor<uint32_t> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(uint32_t);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(dst, src,
                    0,  // sid
                    nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(dst, src,
                    0,  // sid
                    nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(
    AscendC::GlobalTensor<float> dst, 
    AscendC::LocalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(dst, src,
                   0,  // sid
                   nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void colwise_mul_compute_aiv(
    AscendC::GlobalTensor<float> gm_in,
    AscendC::GlobalTensor<float> gm_out,
    AscendC::LocalTensor<float> ub_in,
    AscendC::LocalTensor<float> ub_out,
    AscendC::LocalTensor<uint32_t> ub_offset,
    float s_real, float s_imag, uint32_t copy_len, uint32_t len, uint32_t event_id)
{
    uint32_t repeatTime = (len + 63) / 64;         // 3
    uint32_t computeRepeat = (len / 2 + 63) / 64;  // 2

    uint32_t real_offset = 0;
    uint32_t imag_offset = 32 * 1024 / sizeof(float) / 2;

    AscendC::LocalTensor<float> ub_out_real = ub_out;
    AscendC::LocalTensor<float> ub_out_imag = ub_out[imag_offset];

    copy_vec_gm2ub(ub_in, gm_in, copy_len);

    SET_FLAG(MTE2, V, event_id);
    WAIT_FLAG(MTE2, V, event_id);

    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;

    AscendC::GatherMask<float>(ub_out_real, ub_in, 1, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(ub_out_imag, ub_in, 2, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    PIPE_BARRIER(V);

    // R * R
    muls_v<ArchType::ASCEND_V220, float>(ub_in, ub_out_real, s_real, computeRepeat, 1, 1, 8, 8);

    // R * I
    muls_v<ArchType::ASCEND_V220, float>(ub_in[imag_offset], ub_out_real, s_imag, computeRepeat, 1, 1, 8, 8);

    // I * I
    muls_v<ArchType::ASCEND_V220, float>(ub_out_real, ub_out_imag, s_imag, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * R - I * I
    sub_v<ArchType::ASCEND_V220, float>(ub_in, ub_in, ub_out_real, computeRepeat, 1, 1, 1, 8, 8, 8);

    // I * R
    muls_v<ArchType::ASCEND_V220, float>(ub_out_imag, ub_out_imag, s_real, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * I + I * R
    add_v<ArchType::ASCEND_V220, float>(
        ub_in[imag_offset], ub_out_imag, ub_in[imag_offset], computeRepeat, 1, 1, 1, 8, 8, 8);

    PIPE_BARRIER(V);

    AscendC::Gather(ub_out, ub_in, ub_offset, (uint32_t)0, repeatTime * 64);
    PIPE_BARRIER(ALL);

    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    copy_vec_ub2gm(gm_out, ub_out, copy_len);
}

__aicore__ __inline__ __attribute__((always_inline)) void colwise_mul_aiv(
    AscendC::GlobalTensor<float> gm_in,
    AscendC::GlobalTensor<float> gm_vec,
    AscendC::GlobalTensor<uint32_t> gm_aug,
    AscendC::GlobalTensor<float> gm_out,
    uint32_t m, uint32_t cal_num, uint32_t offset, uint32_t row_num)
{
    // ub 192kb
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_out_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(0 * 1024);
    AscendC::LocalTensor<float> ub_out_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(32 * 1024);
    AscendC::LocalTensor<float> ub_in_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 1024);
    AscendC::LocalTensor<float> ub_in_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);
    AscendC::LocalTensor<uint32_t> ub_offset = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(128 * 1024);

    uint32_t ping_flag = 1;

    uint32_t maxDataCount = 32 * 1024 / sizeof(float);

    uint32_t repeatTime = cal_num / maxDataCount;
    uint32_t remainNum = cal_num % maxDataCount;

    float s_real;
    float s_imag;

    // prepare offset
    copy_vec_gm2ub_uint32(ub_offset, gm_aug, maxDataCount);

    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);

    uint32_t curr_offset = offset;
    if (cal_num == 0) {
        return;
    }
    uint32_t curr_row = curr_offset / cal_num;

    for (uint32_t row_idx = 0; row_idx < row_num; row_idx++) {
        curr_offset = offset + cal_num * row_idx;

        s_real = gm_vec.GetValue((curr_row + row_idx) * 2);
        s_imag = gm_vec.GetValue((curr_row + row_idx) * 2 + 1);

        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        SET_FLAG(S, V, EVENT_ID1);
        WAIT_FLAG(S, V, EVENT_ID1);

        if (repeatTime > 0) {
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            for (uint32_t i = 0; i < repeatTime; i++) {
                auto ub_in = ping_flag ? ub_in_ping : ub_in_pong;
                auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;

                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                WAIT_FLAG(MTE3, MTE2, event_id);

                colwise_mul_compute_aiv(gm_in[curr_offset], gm_out[curr_offset], ub_in, ub_out, ub_offset, s_real,
                                        s_imag, maxDataCount, maxDataCount, event_id);

                SET_FLAG(MTE3, MTE2, event_id);

                curr_offset += maxDataCount;
                ping_flag = 1 - ping_flag;
            }
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        }

        if (remainNum > 0) {
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            auto ub_in = ping_flag ? ub_in_ping : ub_in_pong;
            auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, event_id);

            colwise_mul_compute_aiv(gm_in[curr_offset], gm_out[curr_offset], ub_in, ub_out, ub_offset, s_real, s_imag,
                                    remainNum, remainNum, event_id);

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
            curr_offset += remainNum;
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        }
    }
    PIPE_BARRIER(ALL);
}

__global__ __aicore__ void colwise_mul(GM_ADDR mat, GM_ADDR vec,
                                       GM_ADDR aug, GM_ADDR result,
                                       GM_ADDR workSpace, GM_ADDR tilingGm)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    // AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto core_idx = AscendC::GetBlockIdx();

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    uint32_t m = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));      // num of float elements
    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));  // num of float elements

    uint32_t offset = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8 + 4 * core_idx));
    uint32_t row_num = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8 + 40 * 4 + 4 * core_idx));

    if (row_num <= 0)
        return;

    AscendC::GlobalTensor<float> mat_tensor;
    AscendC::GlobalTensor<float> vec_tensor;
    AscendC::GlobalTensor<uint32_t> aug_tensor;
    AscendC::GlobalTensor<float> result_tensor;

    mat_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mat));
    vec_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vec));
    aug_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(aug));
    result_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(result));
    colwise_mul_aiv(mat_tensor, vec_tensor, aug_tensor, result_tensor, m, n, offset, row_num);
}

// Wrapper function for host to call
void colwise_mul_kernel_do(GM_ADDR mat, GM_ADDR vec, GM_ADDR aug, GM_ADDR result, 
                           GM_ADDR workSpace, GM_ADDR tilingGm,
                           uint32_t numBlocks, void *stream)
{
    colwise_mul<<<numBlocks, nullptr, stream>>>(mat, vec, aug, result, workSpace, tilingGm);
}

