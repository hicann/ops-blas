/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_PARAM_H
#define STRMM_PARAM_H

#include <string>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct StrmmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int64_t m = 0;
    int64_t n = 0;
    float   alpha = 1.0f;
    bool    nullAlpha = false;
    bool    nullA = false;
    bool    nullB = false;
    int64_t lda = 0;
    int64_t ldb = 0;

    StrmmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side   = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo   = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        diag   = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        m      = static_cast<int64_t>(std::stoll(ReadMap(map, "m", "0")));
        n      = static_cast<int64_t>(std::stoll(ReadMap(map, "n", "0")));

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr");
        alpha = nullAlpha ? 0.0f : parseFloat(alphaStr, 1.0f);

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");

        lda = static_cast<int64_t>(std::stoll(ReadMap(map, "lda", "0")));
        ldb = static_cast<int64_t>(std::stoll(ReadMap(map, "ldb", "0")));
    }
};

#endif // STRMM_PARAM_H
