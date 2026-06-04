/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STPMV_PARAM_H
#define STPMV_PARAM_H

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct StpmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int n = 0;
    BlasDataFill ap = BlasDataFill::INDEX;
    BlasDataFill x = BlasDataFill::INDEX;
    int incx = 1;

    StpmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        n = parseInt(ReadMap(map, "n", "0"));
        ap = parseDataFill(ReadMap(map, "aPacked", "INDEX"));
        x = parseDataFill(ReadMap(map, "x", "INDEX"));
        incx = parseInt(ReadMap(map, "incx", "1"));
    }
};

#endif // STPMV_PARAM_H
