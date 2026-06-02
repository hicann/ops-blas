/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRSV_PARAM_H
#define TRSV_PARAM_H

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct TrsvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int64_t n = 0;
    BlasDataFill A = BlasDataFill::RANDOM;
    int64_t lda = 0;
    BlasDataFill x = BlasDataFill::RANDOM;
    int64_t incx = 1;

    TrsvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        n = parseInt(ReadMap(map, "n", "0"));
        A = parseDataFill(ReadMap(map, "A", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(int64_t(1), n))));
        x = parseDataFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
    }
};

#endif // TRSV_PARAM_H
