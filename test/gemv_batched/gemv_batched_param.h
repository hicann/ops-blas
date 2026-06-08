/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMV_BATCHED_PARAM_H
#define GEMV_BATCHED_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct GemvBatchedParam : public BlasTestParamBase {
    // 0=HSH(F16→F16) 1=S(F32) 2=HSS(F16→F32) 3=TST(BF16→BF16) 4=TSS(BF16→F32)
    int dtype = 1;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int batchCount = 1;
    float alpha = 1.0f;
    BlasDataFill alphaFill = BlasDataFill::RANDOM;
    BlasDataFill a = BlasDataFill::RANDOM;
    int lda = 0;
    BlasDataFill x = BlasDataFill::RANDOM;
    int incx = 1;
    float beta = 0.0f;
    BlasDataFill betaFill = BlasDataFill::RANDOM;
    BlasDataFill y = BlasDataFill::RANDOM;
    int incy = 1;

    GemvBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        dtype = parseInt(ReadMap(map, "dtype", "1"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        batchCount = parseInt(ReadMap(map, "batchCount", "1"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        alphaFill = parseDataFill(ReadMap(map, "alpha_fill", "RANDOM"));
        a = parseDataFill(ReadMap(map, "a", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        x = parseDataFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        betaFill = parseDataFill(ReadMap(map, "beta_fill", "RANDOM"));
        y = parseDataFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};

#endif // GEMV_BATCHED_PARAM_H
