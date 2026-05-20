/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file gbmv_test_utils.h
 * \brief Shared utilities for GBMV test programs.
 */

#ifndef GBMV_TEST_UTILS_H
#define GBMV_TEST_UTILS_H

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// aclblasSgbmv is declared in cann_ops_blas.h.

static void fill_banded_matrix(float *A, int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t lda,
    std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::fill(A, A + static_cast<size_t>(lda) * static_cast<size_t>(n), 0.0f);
    for (int64_t j = 0; j < n; j++) {
        int64_t iStart = (j > ku) ? (j - ku) : 0;
        int64_t iEnd = (j + kl < m - 1) ? (j + kl) : (m - 1);
        for (int64_t i = iStart; i <= iEnd; i++) {
            int64_t bandedRow = ku + i - j;
            A[bandedRow + j * lda] = dist(rng);
        }
    }
}

static float *fill_strided_vector(float *v, int64_t len, int64_t inc, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    int64_t absInc = (inc < 0) ? -inc : inc;
    std::fill(v, v + static_cast<size_t>((len - 1) * absInc + 1), 0.0f);
    for (int64_t i = 0; i < len; i++) {
        v[i * absInc] = dist(rng);
    }
    return (inc < 0) ? (v + (len - 1) * absInc) : v;
}

static void fill_contiguous_vector(float *v, int64_t len, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int64_t i = 0; i < len; i++) {
        v[i] = dist(rng);
    }
}

#endif  // GBMV_TEST_UTILS_H
