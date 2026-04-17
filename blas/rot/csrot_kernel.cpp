/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CSROT_KERNEL_H
#define CSROT_KERNEL_H

#include "kernel_operator.h"
#include "../common/common.h"
#include "../common/iterator.h"
#include "../common/simd.h"
#include "../common/utils.h"

using namespace AscendC;

__global__ __aicore__ void csrot_kernel(GM_ADDR gm_x, GM_ADDR gm_y, 
                                        GM_ADDR gm_workspace, GM_ADDR tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    
    // Init
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();

    const int32_t elementCount = *reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    float cosValue = *reinterpret_cast<__gm__ float *>(tiling_para_gm + 4);
    float sinValue = *reinterpret_cast<__gm__ float *>(tiling_para_gm + 8);
    float negaSinValue = 0 - sinValue;

    const int16_t aiv_id = AscendC::GetBlockIdx();
    int16_t aiv_num = AscendC::GetBlockNum();
    if (aiv_num == 0) {
        aiv_num = 1;
    }

    int32_t elementPerCore = elementCount / aiv_num;
    const int32_t elementRemain = elementCount % aiv_num;

    int32_t startIndex = aiv_id * elementPerCore;
    if (aiv_id < elementRemain) {
        startIndex += aiv_id;
        elementPerCore += 1;
    } else {
        startIndex += elementRemain;
    }

    AscendC::GlobalTensor<float> gm_x_tensor;
    AscendC::GlobalTensor<float> gm_y_tensor;
    gm_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_x));
    gm_y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_y));

    
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> buf0_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);               // 24 KB
    AscendC::LocalTensor<float> buf0_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(24 * 1024);       // 24 KB
    AscendC::LocalTensor<float> buf0_temp1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(48 * 1024);   // 24 KB
    AscendC::LocalTensor<float> buf0_temp2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(72 * 1024);   // 24 KB
    AscendC::LocalTensor<float> buf1_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);       // 24 KB
    AscendC::LocalTensor<float> buf1_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(120 * 1024);      // 24 KB
    AscendC::LocalTensor<float> buf1_temp1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(144 * 1024);  // 24 KB
    AscendC::LocalTensor<float> buf1_temp2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(168 * 1024);  // 24 KB

    const int32_t maxElementSingleCount = 24 * 1024 / sizeof(float);
    int32_t loop_count = elementPerCore / maxElementSingleCount;
    const int32_t loop_remain = elementPerCore % maxElementSingleCount;
    if (loop_remain > 0) {
        loop_count += 1;
    }

    bool flag = 0;

    AscendC::GlobalTensor<float> current_gm_x = gm_x_tensor[startIndex];
    AscendC::GlobalTensor<float> current_gm_y = gm_y_tensor[startIndex];

    int32_t currentElementSingleCount = maxElementSingleCount;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
        currentElementSingleCount = (loop_idx == loop_count - 1 && loop_remain > 0) ? loop_remain :
                                                                                      maxElementSingleCount;
        int32_t currentRepeatCount = (currentElementSingleCount + 63) / 64;

        auto buf_x = flag ? buf0_x : buf1_x;
        auto buf_y = flag ? buf0_y : buf1_y;
        auto buf_temp1 = flag ? buf0_temp1 : buf1_temp1;
        auto buf_temp2 = flag ? buf0_temp2 : buf1_temp2;

        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(MTE3, MTE2, event_id);
        // CopyIn
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            buf_x,                                      //__ubuf__ void *dst,
            current_gm_x,                               //__gm__ void *src,
            0,                                          // uint8_t sid,
            1,                                          // uint16_t nBurst,
            currentElementSingleCount * sizeof(float),  // uint32_t lenBurst,
            0,                                          // uint8_t leftPaddingNum,
            0,                                          // uint8_t rightPaddingNum,
            0,                                          // uint32_t srcGap,
            0                                           // uint32_t dstGap
        );

        SET_FLAG(MTE2, V, EVENT_ID0);

        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            buf_y,                                      //__ubuf__ void *dst,
            current_gm_y,                               //__gm__ void *src,
            0,                                          // uint8_t sid,
            1,                                          // uint16_t nBurst,
            currentElementSingleCount * sizeof(float),  // uint32_t lenBurst,
            0,                                          // uint8_t leftPaddingNum,
            0,                                          // uint8_t rightPaddingNum,
            0,                                          // uint32_t srcGap,
            0                                           // uint32_t dstGap
        );

        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        // Compute
        muls_v<ArchType::ASCEND_V220, float>(
            buf_temp1,           // __ubuf__ float *dst,
            buf_x,               // __ubuf__ float *src0,
            cosValue,            // float src1,
            currentRepeatCount,  // uint8_t repeat,
            1,                   // uint16_t dstBlockStride,
            1,                   // uint16_t srcBlockStride,
            8,                   // uint8_t dstRepeatStride,
            8                    // uint8_t srcRepeatStride
        );
        muls_v<ArchType::ASCEND_V220, float>(\
            buf_temp2,           // __ubuf__ float *dst,
            buf_x,               // __ubuf__ float *src0,
            negaSinValue,        // float src1,
            currentRepeatCount,  // uint8_t repeat,
            1,                   // uint16_t dstBlockStride,
            1,                   // uint16_t srcBlockStride,
            8,                   // uint8_t dstRepeatStride,
            8                    // uint8_t srcRepeatStride
        );

        PIPE_BARRIER(V);

        WAIT_FLAG(MTE2, V, EVENT_ID1);

        AscendC::Axpy(buf_temp1, buf_y, sinValue, currentRepeatCount * 64);
        AscendC::Axpy(buf_temp2, buf_y, cosValue, currentRepeatCount * 64);

        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        // CopyOut
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            current_gm_x,                               //__gm__ void *dst,
            buf_temp1,                                  //__ubuf__ void *src,
            0,                                          // uint8_t sid,
            1,                                          // uint16_t nBurst,
            currentElementSingleCount * sizeof(float),  // uint32_t lenBurst,
            0,                                          // uint8_t leftPaddingNum,
            0,                                          // uint8_t rightPaddingNum,
            0,                                          // uint32_t srcGap,
            0                                           // uint32_t dstGap
        );

        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            current_gm_y,                               //__gm__ void *dst,
            buf_temp2,                                  //__ubuf__ void *src,
            0,                                          // uint8_t sid,
            1,                                          // uint16_t nBurst,
            currentElementSingleCount * sizeof(float),  // uint32_t lenBurst,
            0,                                          // uint8_t leftPaddingNum,
            0,                                          // uint8_t rightPaddingNum,
            0,                                          // uint32_t srcGap,
            0                                           // uint32_t dstGap
        );

        SET_FLAG(MTE3, MTE2, event_id);
        current_gm_x = current_gm_x[maxElementSingleCount];
        current_gm_y = current_gm_y[maxElementSingleCount];
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

void csrot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream)
{
    csrot_kernel<<<numBlocks, nullptr, stream>>>(x, y, workSpace, tilingGm);
}

#endif  // CSROT_KERNEL_H
