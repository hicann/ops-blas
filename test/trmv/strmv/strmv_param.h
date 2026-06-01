/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRMV_PARAM_H
#define TRMV_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct StrmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int n = 0;
    int lda = 0;
    int incx = 1;

    StrmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        std::string uploStr = ReadMap(map, "uplo", "LOWER");
        std::string transStr = ReadMap(map, "trans", "N");
        std::string diagStr = ReadMap(map, "diag", "NON_UNIT");

        if (uploStr == "INVALID") {
            uplo = static_cast<aclblasFillMode_t>(0xFF);
        } else {
            uplo = parseFillMode(uploStr);
        }
        if (transStr == "INVALID") {
            trans = static_cast<aclblasOperation_t>(0xFF);
        } else {
            trans = parseOpTrans(transStr);
        }
        if (diagStr == "INVALID") {
            diag = static_cast<aclblasDiagType_t>(0xFF);
        } else {
            diag = parseDiagType(diagStr);
        }

        n = parseInt(ReadMap(map, "n", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(n)));
        incx = parseInt(ReadMap(map, "incx", "1"));
    }
};

#endif // TRMV_PARAM_H
