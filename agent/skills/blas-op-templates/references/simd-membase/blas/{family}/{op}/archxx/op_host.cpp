/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMD membase Host 侧实现
// 标准骨架：参数校验 → 获取核数 → 计算 Tiling → 分配设备内存 → 启动 kernel → 释放

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "{{op}}_tiling_data.h"

void {{op}}_kernel_do(/* TEMPLATE: 参数与 kernel entry / _do 签名一致 */);


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
// - 按算子 API 的参数列表自定义校验逻辑
// - 校验顺序：nullptr → 零值 → 非法值
// - 使用 OP_LOGE 输出错误信息
static aclblasStatus_t Validate{{Op}}Params(/* 算子的参数列表 */)
{
    // TEMPLATE: 按算子需求校验各参数
    // if (ptr == nullptr) { OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE; }
    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: Tiling 计算函数
// - 在 host 侧计算多核切分参数，填充 TilingData 结构体
// - UB tile 大小一般按 UB_SIZE / buffer_count / sizeof(dtype) 计算并对齐
static {{Op}}TilingData Cal{{Op}}TilingData(/* 算子的维度参数, */ uint32_t coreNum)
{
    {{Op}}TilingData tiling{};
    // TEMPLATE: 按算子的切分策略填充 tiling 各字段
    return tiling;
}

// TEMPLATE: 公共 API 入口
// - 签名与 cann_ops_blas.h 中的声明一致
// - 前置快速返回：n<=0 → SUCCESS，handle==nullptr → HANDLE_IS_NULLPTR
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle /* , 算子参数 */)
{
    // TEMPLATE: 快速返回条件（按算子语义调整）
    // if (n <= 0) return ACLBLAS_STATUS_SUCCESS;

    if (handle == nullptr) {
        OP_LOGE("aclblas{{Op}}", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // TEMPLATE: 参数校验
    aclblasStatus_t st = Validate{{Op}}Params(/* 参数 */);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    // TEMPLATE: 确定核数（按算子维度与核数的关系）
    uint32_t numBlocks = /* min(totalWork, aivCoreNum) */;

    {{Op}}TilingData tiling = Cal{{Op}}TilingData(/* 维度参数, */ numBlocks);

    OP_LOGD("aclblas{{Op}}", "tiling: ...");
    OP_LOGI("aclblas{{Op}}", "launching kernel");

    // TEMPLATE: 如有 workspace 需求，额外 aclrtMalloc 分配
    // 分配 Tiling 设备内存
    uint8_t* tilingDevice = nullptr;
    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof({{Op}}TilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblas{{Op}}", "aclrtMalloc failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(
        tilingDevice, sizeof({{Op}}TilingData), &tiling, sizeof({{Op}}TilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblas{{Op}}", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // TEMPLATE: 启动 kernel（参数与 kernel entry / _do 签名匹配）
    {{op}}_kernel_do(/* GM_ADDR 各参数, */ tilingDevice, numBlocks, useStream);

    aclrtFree(tilingDevice);
    // TEMPLATE: 如有 workspace，也需 aclrtFree

    return ACLBLAS_STATUS_SUCCESS;
}
