/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "{{op}}_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "tiling/platform/platform_ascendc.h"

void {{op}}_kernel_do(/* TEMPLATE: 参数与 kernel entry / _do 签名一致 */);


// TEMPLATE: UB 总大小
// 注意：kernel_constant.h 中定义 UB_SIZE = 248 * 1024（理论最大值）
// 但实际可用 UB 通常小于此值（需预留空间给栈、临时变量等）
// 建议根据算子复杂度选择：
//   - 简单算子：190 * 1024（190KB，保守值）
//   - 复杂算子：240 * 1024（240KB，接近最大值）
constexpr uint32_t UB_SIZE = 190 * 1024;  // 190KB for typical Ascend chip

// TEMPLATE: 对齐常量
constexpr uint32_t ALIGN_BYTES = 256;  // UB buffer 需要 256B 对齐

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

static uint32_t AlignUp(uint32_t size, uint32_t align)
{
    return (size + align - 1) & ~(align - 1);
}

// TEMPLATE: 计算 UB buffer 布局
struct BufferLayout {
    uint32_t bufInput;
    uint32_t bufOutput;
    uint32_t bufTemp;
    uint32_t total;
};

static BufferLayout ComputeBufferLayout(uint32_t m, uint32_t n)
{
    BufferLayout layout = {};
    
    // TEMPLATE: 根据算子需求计算各 buffer 大小（256B 对齐）
    // layout.bufInput = AlignUp(m * n * sizeof(float), ALIGN_BYTES);
    // layout.bufOutput = AlignUp(m * sizeof(float), ALIGN_BYTES);
    // layout.bufTemp = AlignUp(m * sizeof(float), ALIGN_BYTES);
    
    layout.total = layout.bufInput + layout.bufOutput + layout.bufTemp;
    return layout;
}

static {{Op}}TilingData CalTilingData(
    /* 算子维度参数, 标量参数, */
    uint32_t coreNum, uint32_t usedCoreNum, uint32_t batchPerCore)
{
    {{Op}}TilingData tiling = {};
    
    // TEMPLATE: 填充基础参数
    // tiling.m = m;
    // tiling.n = n;
    // tiling.alpha = alpha;
    // tiling.beta = beta;
    
    // TEMPLATE: 填充多核切分参数
    tiling.coreNum = coreNum;
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = /* batchCount - (usedCoreNum - 1) * batchPerCore */;
    
    // TEMPLATE: 计算 UB buffer 大小
    // auto layout = ComputeBufferLayout(/* 维度参数 */);
    // tiling.bufInput = layout.bufInput;
    // tiling.bufOutput = layout.bufOutput;
    // tiling.bufTemp = layout.bufTemp;
    
    return tiling;
}

aclblasStatus_t aclblas{{Op}}(
    aclblasHandle_t handle
    // TEMPLATE: 添加算子特定参数（如维度、标量、指针等）
)
{
    // TEMPLATE: 参数校验
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    // TEMPLATE: 校验其他参数（维度、指针等）
    
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;
    
    // TEMPLATE: 计算多核切分
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "aiv core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    
    // TEMPLATE: 根据算子维度计算 batch 分配
    // uint32_t batchPerCore = (batchCount + coreNum - 1) / coreNum;
    // uint32_t usedCoreNum = (batchCount + batchPerCore - 1) / batchPerCore;
    
    {{Op}}TilingData tiling = CalTilingData(
        // TEMPLATE: 传入算子特定参数
        coreNum /*, usedCoreNum, batchPerCore */);
    
    // TEMPLATE: 分配 workspace（如需要）
    uint8_t* workSpaceDevice = nullptr;
    // aclError aclRet = aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    // TEMPLATE: 分配 tiling device 内存
    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof({{Op}}TilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        if (workSpaceDevice) aclrtFree(workSpaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    
    // TEMPLATE: 拷贝 tiling 到 device
    aclRet = aclrtMemcpy(tilingDevice, sizeof({{Op}}TilingData), &tiling, sizeof({{Op}}TilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(tilingDevice);
        if (workSpaceDevice) aclrtFree(workSpaceDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // TEMPLATE: 启动 kernel
    {{op}}_kernel_do(
        // TEMPLATE: 传入算子特定的 GM 指针
        workSpaceDevice, tilingDevice,
        usedCoreNum, useStream);
    
    // TEMPLATE: 释放资源
    aclrtFree(tilingDevice);
    if (workSpaceDevice) aclrtFree(workSpaceDevice);
    
    return ACLBLAS_STATUS_SUCCESS;
}
