/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STPTTR_PARAM_H
#define STPTTR_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct StpttrParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int n = 0;
    BlasDataFill ap = BlasDataFill::INDEX;
    BlasDataFill a = BlasDataFill::SENTINEL;
    int lda = 0;

    StpttrParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        n = parseInt(ReadMap(map, "n", "0"));
        ap = parseDataFill(ReadMap(map, "ap", "INDEX"));
        a = parseDataFill(ReadMap(map, "a", "SENTINEL"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
    }
};

#endif // STPTTR_PARAM_H
