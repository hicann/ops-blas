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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cann_ops_blas.h"
#include "types.h"

constexpr float kStpttrSentinel = -999.0f;

enum StpttrSpecialValueType { STPTTR_NORMAL, STPTTR_ZEROS, STPTTR_LARGE, STPTTR_NEGATIVE,
                              STPTTR_INF, STPTTR_NAN_VAL, STPTTR_EXTREME };

static aclblasFillMode_t parseUplo(const std::string& uplo) {
    if (uplo == "UPPER" || uplo == "U") return ACLBLAS_UPPER;
    if (uplo == "LOWER" || uplo == "L") return ACLBLAS_LOWER;
    if (uplo == "INVALID" || uplo == "0xFF") return static_cast<aclblasFillMode_t>(0xFF);
    try {
        return static_cast<aclblasFillMode_t>(std::stoi(uplo));
    } catch (...) {
        return static_cast<aclblasFillMode_t>(0xFF);
    }
}

static StpttrSpecialValueType specialValueTypeFromDescription(const std::string& description) {
    if (description.find("zeros") != std::string::npos) return STPTTR_ZEROS;
    if (description.find("large") != std::string::npos) return STPTTR_LARGE;
    if (description.find("neg") != std::string::npos) return STPTTR_NEGATIVE;
    if (description.find("inf") != std::string::npos) return STPTTR_INF;
    if (description.find("nan") != std::string::npos) return STPTTR_NAN_VAL;
    if (description.find("extr") != std::string::npos) return STPTTR_EXTREME;
    return STPTTR_NORMAL;
}

static std::vector<float> makeStpttrApData(int n, StpttrSpecialValueType svt) {
    size_t apSize = static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2;
    std::vector<float> ap(apSize);
    switch (svt) {
    case STPTTR_ZEROS:
        std::fill(ap.begin(), ap.end(), 0.0f);
        break;
    case STPTTR_LARGE:
        std::fill(ap.begin(), ap.end(), 1e10f);
        break;
    case STPTTR_NEGATIVE:
        for (size_t i = 0; i < apSize; i++) {
            ap[i] = -static_cast<float>(i + 1);
        }
        break;
    case STPTTR_INF:
        std::fill(ap.begin(), ap.end(), INFINITY);
        break;
    case STPTTR_NAN_VAL:
        std::fill(ap.begin(), ap.end(), NAN);
        break;
    case STPTTR_EXTREME: {
        const float v[] = {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
        for (size_t i = 0; i < apSize; i++) {
            ap[i] = v[i % 7];
        }
        break;
    }
    default:
        for (size_t i = 0; i < apSize; i++) {
            ap[i] = static_cast<float>(i + 1);
        }
        break;
    }
    return ap;
}

// Reference unpack: AP (packed triangular) -> full lda-by-n matrix A.
static void stpttr_golden_unpack(int n, aclblasFillMode_t uplo,
                                 const float* ap, int lda, float* a) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < lda; i++) {
            a[j * lda + i] = kStpttrSentinel;
        }
    }
    int idx = 0;
    if (uplo == ACLBLAS_LOWER) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                a[j * lda + i] = ap[idx++];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                a[j * lda + i] = ap[idx++];
            }
        }
    }
}

static void stpttr_golden_impl(const TestCaseConfig& tc, aclblasFillMode_t uplo,
                               const std::vector<float>& ap,
                               std::vector<float>& goldenOutput) {
    const int n = static_cast<int>(tc.n.value_or(0));
    const int lda = static_cast<int>(tc.lda.value_or(n));
    goldenOutput.assign(static_cast<size_t>(lda * n), kStpttrSentinel);
    if (n > 0) {
        stpttr_golden_unpack(n, uplo, ap.data(), lda, goldenOutput.data());
    }
}
