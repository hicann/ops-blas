/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMT arch35 Host 侧实现
// 与 SIMD membase 的区别：
//   - 需额外计算 nthreads（每 block 线程数）
//   - block 数按 CeilDiv(n, SIMT_MIN_THREAD_NUM) 与 coreNum 取小
//   - tiling 中包含标量参数（alpha/beta/uplo 等）直接传给 kernel

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "tiling/platform/platform_ascendc.h"
#include "{{op}}_tiling_data.h"

// 获取 AIV Core 数量（通过 PlatformAscendCManager，比 aclrtGetDeviceInfo 更可靠）
static uint32_t GetAivCoreCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("GetAivCoreCount", "PlatformAscendCManager::GetInstance() returned nullptr");
        return 0;
    }
    return platform->GetCoreNumAiv();
}

// TEMPLATE: 参数校验函数
static aclblasStatus_t Validate{{Op}}Params(/* 算子的参数列表 */)
{
    // TEMPLATE: 按算子需求校验各参数
    // CHECK_RET(ptr != nullptr, OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: Tiling 计算函数
// SIMT 特有：计算 nthreads
//   nthreads = min(CeilAlign(CeilDiv(totalWork, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)
static {{Op}}TilingData Cal{{Op}}TilingData(uint32_t useNumBlocks /* , 算子参数 */)
{
    {{Op}}TilingData tiling{};
    // TEMPLATE: 计算 nthreads
    // tiling.nthreads = std::min(
    //     CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalWork, useNumBlocks), SIMT_MIN_THREAD_NUM),
    //     SIMT_MAX_THREAD_NUM);
    // TEMPLATE: 填充其他字段（维度参数 + 标量参数）
    return tiling;
}

// TEMPLATE: 公共 API 入口
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle /* , 算子参数 */)
{
    // TEMPLATE: 快速返回条件（按算子语义调整）
    // CHECK_RET(n >= 0, OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE);
    // if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    CHECK_RET(handle != nullptr, OP_LOGE("aclblas{{Op}}", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    aclblasStatus_t st = Validate{{Op}}Params(/* 参数 */);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // TEMPLATE: SIMT 的 block 数计算（与 SIMD membase 不同）
    // useNumBlocks = min(CeilDiv(totalWork, SIMT_MIN_THREAD_NUM), aivCoreNum)
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(/* totalWork */, SIMT_MIN_THREAD_NUM), aivCoreNum);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    {{Op}}TilingData tiling = Cal{{Op}}TilingData(useNumBlocks /* , 参数 */);

    OP_LOGD("aclblas{{Op}}", "tiling: nthreads=%u numBlocks=%u", tiling.nthreads, useNumBlocks);
    OP_LOGI("aclblas{{Op}}", "launching kernel");

    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof({{Op}}TilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblas{{Op}}", "aclrtMalloc failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(
        tilingDevice, sizeof({{Op}}TilingData), &tiling, sizeof({{Op}}TilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, OP_LOGE("aclblas{{Op}}", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    // TEMPLATE: 启动 kernel
    {{op}}_kernel_do(/* GM_ADDR 各参数, */ nullptr, tilingDevice, useNumBlocks, useStream);

    // TEMPLATE: 按需决定是否同步 stream
    // 大多数 BLAS 算子在 stream 上异步执行，同步由上层调用方负责
    // 如算子内部有 host 侧数据依赖（如读取 workspace 中间结果），则需要同步：
    // aclRet = aclrtSynchronizeStream(useStream);
    // CHECK_RET(aclRet == ACL_SUCCESS, OP_LOGE(...); aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
