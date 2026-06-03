/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYR2_PARAM_H
#define SSYR2_PARAM_H

#include <algorithm>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct Ssyr2Param : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alpha = 1.0f;
    BlasDataFill x = BlasDataFill::RANDOM;
    int incx = 1;
    BlasDataFill y = BlasDataFill::RANDOM;
    int incy = 1;
    BlasDataFill a = BlasDataFill::RANDOM;
    int lda = 0;

    Ssyr2Param(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        x = parseDataFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseDataFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        a = parseDataFill(ReadMap(map, "a", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
    }
};

#endif // SSYR2_PARAM_H
