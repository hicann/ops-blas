/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use the License for the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IAMAX_KERNEL_H
#define IAMAX_KERNEL_H

// Include the full implementation from the original file
// The implementation has been migrated from /opt/nzh/sip/ops/blas/iamax/iamax/kernel/iamax.h

#include "iamax_kernel_impl.h"

// Kernel entry point
__global__ __aicore__ void iamax_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = workspace;
    Iamax::Iamax<float> op;
    op.Init(x, y, userWS, tiling);
    op.Process();
}

// Wrapper function for host to call
void iamax_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                     uint32_t numBlocks, void *stream)
{
    iamax_kernel<<<numBlocks, nullptr, stream>>>(x, y, workspace, tiling);
}

#endif  // IAMAX_KERNEL_H
