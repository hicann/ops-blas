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
 * \file srotm_host.cpp
 * \brief Modified Givens rotation host implementation
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"
#include "srotm_tiling_data.h"

void srotm_kernel_do_arch35(
    uint8_t* x, uint8_t* y, const SrotmTilingData& tilingData, uint32_t numBlocks, void *stream);

static void BuildTilingData(int n, int incx, int incy, const float *param,
                            uint32_t numBlocks, SrotmTilingData &tilingData)
{
    const float flag = param[0];
    float alpha1, beta1, alpha2, beta2;
    if (flag == -1.0f) {
        alpha1 = param[1];
        beta1 = param[3];
        alpha2 = param[2];
        beta2 = param[4];
    } else if (flag == 0.0f) {
        alpha1 = 1.0f;
        beta1 = param[3];
        alpha2 = param[2];
        beta2 = 1.0f;
    } else {
        alpha1 = param[1];
        beta1 = 1.0f;
        alpha2 = -1.0f;
        beta2 = param[4];
    }

    tilingData.elementCount = static_cast<int32_t>(n);
    tilingData.alpha1 = alpha1; tilingData.beta1 = beta1;
    tilingData.alpha2 = alpha2; tilingData.beta2 = beta2;
    tilingData.incx = static_cast<int32_t>(incx);
    tilingData.incy = static_cast<int32_t>(incy);
    tilingData.kx = (incx >= 0) ? 0 : (1LL - n) * incx;
    tilingData.ky = (incy >= 0) ? 0 : (1LL - n) * incy;
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), numBlocks), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);
}

aclblasStatus_t aclblasSrotm(
    aclblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrotm", "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n <= 0) { return ACLBLAS_STATUS_SUCCESS; }
    if (x == nullptr || y == nullptr || param == nullptr) {
        OP_LOGE("aclblasSrotm", "input pointers contain a nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasSrotm", "incx or incy is 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const float flag = param[0];
    if (flag == -2.0f) { return ACLBLAS_STATUS_SUCCESS; }

    auto *h = handle;

    uint32_t numBlocks = GetAivCoreCount();
    if (numBlocks == 0) {
        OP_LOGE("aclblasSrotm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    SrotmTilingData tilingData;
    BuildTilingData(n, incx, incy, param, numBlocks, tilingData);

    srotm_kernel_do_arch35(reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y),
                    tilingData, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
