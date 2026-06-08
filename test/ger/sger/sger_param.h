/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGER_PARAM_H
#define SGER_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct SgerParam : public BlasTestParamBase {
    int m = 0;
    int n = 0;
    BlasDataFill alpha = BlasDataFill::RANDOM;
    float alphaValue = 1.0f;
    BlasDataFill x = BlasDataFill::INDEX;
    int incx = 1;
    BlasDataFill y = BlasDataFill::INDEX;
    int incy = 1;
    BlasDataFill A = BlasDataFill::INDEX;
    int lda = 0;

    SgerParam(const csv_map& map) : BlasTestParamBase(map)
    {
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseDataFill(ReadMap(map, "alpha", "RANDOM"));
        alphaValue = parseFloat(ReadMap(map, "alpha_value", "1.0"));
        x = parseDataFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseDataFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        A = parseDataFill(ReadMap(map, "A", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
    }
};

#endif // SGER_PARAM_H
