/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCBLAS_FP32_UTILS_H
#define ASCBLAS_FP32_UTILS_H

#include "iterator.h"

namespace fp32 {
constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(float);
constexpr int64_t CUBE_M0 = 16;
constexpr int64_t CUBE_N0 = 16;
constexpr int64_t CUBE_K0 = 32 / sizeof(float);
constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

__aicore__ inline int64_t ROUND(int64_t num, int64_t paddingNum)
{
    return ((num + paddingNum - 1) / paddingNum * paddingNum);
}

__aicore__ __inline__ void ascblas_matrix_gm2cbuf_ND2nZ(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int64_t CBUF_M0, int64_t CBUF_N0, int64_t mActual, int64_t nActual, size_t stride)
{
    if (stride < UINT16_STRIDE_LIMIT) {
        AscendC::DataCopy(
            dst, src,
            AscendC::Nd2NzParams(
                1,
                nActual,
                mActual,
                0,
                stride,
                CBUF_N0,
                1,
                0)
        );
    } else {
        for (int i = 0; i < nActual; i++) {
            AscendC::DataCopy(
                dst[i * CUBE_K0], src[i * stride],
                AscendC::Nd2NzParams(
                    1,
                    1,
                    mActual,
                    0,
                    0,
                    CBUF_N0,
                    0,
                    0)
            );
        }
    }
}

__aicore__ __inline__ void ascblas_matrix_gm2cbuf_ND2nN(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int64_t CBUF_M0, int64_t CBUF_N0, int64_t mActual, int64_t nActual, size_t stride)
{
    int64_t srcNdStride = CUBE_N0 * stride;
    int64_t srcNStride = stride;
    if (srcNdStride < UINT16_STRIDE_LIMIT) {
        int ndNum = nActual / CUBE_N0;
        int remains = nActual % CUBE_N0;
        if (ndNum > 0) {
            AscendC::DataCopy(
                dst, src,
                AscendC::Nd2NzParams(
                    ndNum,
                    CUBE_N0,
                    mActual,
                    srcNdStride,
                    srcNStride,
                    CUBE_N0,
                    1,
                    CUBE_N0 * CBUF_M0)
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * CUBE_N0 * CBUF_M0], src[ndNum * CUBE_N0 * stride],
                AscendC::Nd2NzParams(
                    1,
                    remains,
                    mActual,
                    0,
                    srcNStride,
                    CUBE_N0,
                    1,
                    0)
            );
        }
    } else if (srcNStride < UINT16_STRIDE_LIMIT) {
        int ndNum = nActual / CUBE_N0;
        int remains = nActual % CUBE_N0;
        for (int i = 0; i < ndNum; i++) {
            AscendC::DataCopy(
                dst[i * CUBE_N0 * CBUF_M0], src[i * CUBE_N0 * stride],
                AscendC::Nd2NzParams(
                    1,
                    CUBE_N0,
                    mActual,
                    0,
                    srcNStride,
                    CUBE_N0,
                    1,
                    0)
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * CUBE_N0 * CBUF_M0], src[ndNum * CUBE_N0 * stride],
                AscendC::Nd2NzParams(
                    1,
                    remains,
                    mActual,
                    0,
                    srcNStride,
                    CUBE_N0,
                    1,
                    0)
            );
        }
    } else {
        for (int i = 0; i < nActual; i++) {
            int idxR0 = i / CUBE_N0;
            int idxInR0 = i % CUBE_N0;
            AscendC::DataCopy(
                dst[idxR0 * CUBE_N0 * CBUF_M0 + idxInR0 * CUBE_K0], src[i * stride],
                AscendC::Nd2NzParams(
                    1,
                    1,
                    mActual,
                    0,
                    0,
                    CUBE_N0,
                    0,
                    0)
            );
        }
    }
}

__aicore__ __inline__ void ascblas_matrix_gm2ubuf(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int64_t mActual, int64_t nActual, size_t srcStride, size_t dstStride)
{
    int64_t mRound = ROUND(mActual, NUM_ELE_PERBLOCK);
    for (int i = 0; i < nActual; i++) {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            dst[i * dstStride], src[i * srcStride], 0, 1, mActual * sizeof(float), 0, 0, 0, 0);
    }
}

__aicore__ __inline__ void ascblas_matrix_ubuf2gm(
    AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    int64_t mActual, int64_t nActual, size_t srcStride, size_t dstStride)
{
    for (int i = 0; i < nActual; i++) {
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            dst[i * dstStride], src[i * srcStride], 0, 1, mActual * sizeof(float), 0, 0, 0, 0);
    }
}
}
#endif