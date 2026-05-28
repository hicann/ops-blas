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

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cann_ops_blas.h"
#include "types.h"

enum StrttpSpecialValueType {
    STRTTP_NORMAL,
    STRTTP_ZEROS,
    STRTTP_LARGE,
    STRTTP_NEGATIVE,
    STRTTP_INF,
    STRTTP_NAN_VAL,
    STRTTP_EXTREME
};

static aclblasFillMode_t parseStrttpUplo(const std::string& uplo) {
    if (uplo == "UPPER" || uplo == "U") {
        return ACLBLAS_UPPER;
    }
    if (uplo == "LOWER" || uplo == "L") {
        return ACLBLAS_LOWER;
    }
    if (uplo == "INVALID" || uplo == "0xFF") {
        return static_cast<aclblasFillMode_t>(0xFF);
    }
    try {
        return static_cast<aclblasFillMode_t>(std::stoi(uplo));
    } catch (...) {
        return static_cast<aclblasFillMode_t>(0xFF);
    }
}

static StrttpSpecialValueType strttpSpecialValueTypeFromDescription(const std::string& description) {
    if (description.find("zeros") != std::string::npos) {
        return STRTTP_ZEROS;
    }
    if (description.find("large") != std::string::npos) {
        return STRTTP_LARGE;
    }
    if (description.find("neg") != std::string::npos) {
        return STRTTP_NEGATIVE;
    }
    if (description.find("inf") != std::string::npos) {
        return STRTTP_INF;
    }
    if (description.find("nan") != std::string::npos) {
        return STRTTP_NAN_VAL;
    }
    if (description.find("extr") != std::string::npos) {
        return STRTTP_EXTREME;
    }
    return STRTTP_NORMAL;
}

static void fillStrttpDenseMatrix(int n, int lda, StrttpSpecialValueType svt, std::vector<float>& a) {
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    a.assign(aLen, 0.0f);
    switch (svt) {
    case STRTTP_ZEROS:
        return;
    case STRTTP_LARGE:
        std::fill(a.begin(), a.end(), 1e10f);
        return;
    case STRTTP_NEGATIVE:
        for (size_t i = 0; i < aLen; i++) {
            a[i] = -static_cast<float>(i + 1);
        }
        return;
    case STRTTP_INF:
        std::fill(a.begin(), a.end(), INFINITY);
        return;
    case STRTTP_NAN_VAL:
        std::fill(a.begin(), a.end(), NAN);
        return;
    case STRTTP_EXTREME: {
        const float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
        for (size_t i = 0; i < aLen; i++) {
            a[i] = v[i % 7];
        }
        return;
    }
    default:
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                a[static_cast<size_t>(j) * static_cast<size_t>(lda) + static_cast<size_t>(i)] =
                    static_cast<float>(j * n + i + 1);
            }
        }
        return;
    }
}

// Reference pack: dense lda-by-n matrix A -> packed triangular AP.
static void strttp_golden_pack(int n, int lda, aclblasFillMode_t uplo,
                               const float* a, std::vector<float>& ap) {
    ap.assign(static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2, 0.0f);
    size_t idx = 0;
    if (uplo == ACLBLAS_LOWER) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                ap[idx++] = a[static_cast<size_t>(j) * static_cast<size_t>(lda) + static_cast<size_t>(i)];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                ap[idx++] = a[static_cast<size_t>(j) * static_cast<size_t>(lda) + static_cast<size_t>(i)];
            }
        }
    }
}

static void strttp_golden_impl(const TestCaseConfig& tc, aclblasFillMode_t uplo,
                               const std::vector<float>& aHost,
                               std::vector<float>& goldenOutput) {
    const int n = static_cast<int>(tc.n.value_or(0));
    const int lda = static_cast<int>(tc.lda.value_or(n));
    if (n <= 0) {
        goldenOutput.clear();
        return;
    }
    strttp_golden_pack(n, lda, uplo, aHost.data(), goldenOutput);
}
