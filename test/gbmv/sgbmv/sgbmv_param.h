/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct GbmvParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int kl = 0;
    int ku = 0;
    float alpha = 1.0f;
    BlasFillMode a = parseFill("RANDOM");
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    float beta = 0.0f;
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;

    GbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        kl = parseInt(ReadMap(map, "kl", "0"));
        ku = parseInt(ReadMap(map, "ku", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        a = parseFill(ReadMap(map, "a", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(kl + ku + 1)));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};

