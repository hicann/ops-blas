/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ssbmv_kernel.cpp
 * \brief single-precision sbmv SIMT kernel
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ssbmv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t SSBMV_SIMT_MAX_THREAD_NUM = 2048;

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SSBMV_SIMT_MAX_THREAD_NUM) inline void SsbmvSimtCompute(
    uint32_t n, uint32_t k, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float acc = 0.0f;

        uint32_t jStart = (row >= k) ? (row - k) : 0;
        uint32_t jEnd = (n < row + k + 1) ? n : (row + k + 1);

        for (uint32_t jCol = jStart; jCol < jEnd; ++jCol) {
            int64_t gmIdx;
            if constexpr (UPLO_IS_UPPER) {
                if (row <= jCol) {
                    gmIdx = (k + row - jCol) + static_cast<int64_t>(lda) * jCol;
                } else {
                    gmIdx = (k + jCol - row) + static_cast<int64_t>(lda) * row;
                }
            } else {
                if (row >= jCol) {
                    gmIdx = (row - jCol) + static_cast<int64_t>(lda) * jCol;
                } else {
                    gmIdx = (jCol - row) + static_cast<int64_t>(lda) * row;
                }
            }

            float xVal;
            if (incx >= 0) {
                xVal = xGm[jCol * incx];
            } else {
                xVal = xGm[(n - 1 - jCol) * (-incx)];
            }

            acc += aGm[gmIdx] * xVal;
        }

        int64_t yIdx;
        float yVal;
        if (incy >= 0) {
            yIdx = row * incy;
            yVal = yGm[yIdx];
        } else {
            yIdx = (n - 1 - row) * (-incy);
            yVal = yGm[yIdx];
        }
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

__global__ __aicore__ void ssbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const auto* __restrict tdata = reinterpret_cast<__gm__ SsbmvTilingData*>(tilingGm);

    if (tdata->uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsbmvSimtCompute<true>>(
            dim3{SSBMV_SIMT_MAX_THREAD_NUM, 1, 1}, tdata->n, tdata->k, tdata->lda, tdata->alpha, tdata->beta,
            tdata->incx, tdata->incy, reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<SsbmvSimtCompute<false>>(
            dim3{SSBMV_SIMT_MAX_THREAD_NUM, 1, 1}, tdata->n, tdata->k, tdata->lda, tdata->alpha, tdata->beta,
            tdata->incx, tdata->incy, reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y));
    }
}

void ssbmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    ssbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tilingGm);
}
