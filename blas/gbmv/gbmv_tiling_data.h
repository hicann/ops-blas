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
 * \file gbmv_tiling_data.h
 * \brief Shared tiling data structure for GBMV host and kernel.
 */

#ifndef GBMV_TILING_DATA_H
#define GBMV_TILING_DATA_H

#include <cstdint>

template <typename T>
struct GbmvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    uint32_t useCoreNum;
    int32_t trans;
    T alpha;
    T beta;
    uint32_t maxSegLen;
};

#endif  // GBMV_TILING_DATA_H
