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

#include <cstdint>
#include <random>
#include <vector>

#include "data.h"
#include "types.h"
#include "verify.h"

inline int64_t computeLogicalOutputCountGbmv(const TestCaseConfig& tc) {
    const bool isTransN = (!tc.trans.has_value() || tc.trans.value() == "N");
    return isTransN ? tc.m.value_or(0) : tc.n.value_or(0);
}

inline std::vector<std::vector<float>> generateHostDataGbmv(const TestCaseConfig& tc) {
    std::vector<std::vector<float>> hostData;
    const uint32_t seed = tc.seed.has_value() ? tc.seed.value() : 42;

    const int64_t m = tc.m.value_or(8);
    const int64_t n = tc.n.value_or(8);
    const int64_t kl = tc.kl.value_or(2);
    const int64_t ku = tc.ku.value_or(2);
    const int64_t lda = tc.lda.value_or(kl + ku + 1);
    const int64_t incx = tc.incx.value_or(1);
    const int64_t incy = tc.incy.value_or(1);
    const bool isTransN = (!tc.trans.has_value() || tc.trans.value() == "N");
    const int64_t xCount = isTransN ? n : m;
    const int64_t yCount = isTransN ? m : n;
    const int64_t absIncx = (incx < 0) ? -incx : incx;
    const int64_t absIncy = (incy < 0) ? -incy : incy;

    const size_t aSize = (n > 0) ? static_cast<size_t>(lda) * static_cast<size_t>(n) : 1;
    std::vector<float> a(aSize, 0.0f);
    const size_t xSize = (xCount > 0) ? static_cast<size_t>((xCount - 1) * absIncx + 1) : 1;
    std::vector<float> x(xSize, 0.0f);
    const size_t ySize = (yCount > 0) ? static_cast<size_t>((yCount - 1) * absIncy + 1) : 1;
    std::vector<float> y(ySize, 0.0f);

    std::mt19937 rng(seed);
    DataGenerator::fillBandedMatrix(a.data(), m, n, kl, ku, lda, rng);
    DataGenerator::fillStridedVector(x.data(), xCount, incx, rng);
    DataGenerator::fillStridedVector(y.data(), yCount, incy, rng);

    hostData.push_back(std::move(a));
    hostData.push_back(std::move(x));
    hostData.push_back(std::move(y));

    hostData.push_back(hostData[2]);
    hostData.push_back({tc.alphaReal.value_or(1.0f)});
    hostData.push_back({tc.betaReal.value_or(0.0f)});
    return hostData;
}

inline bool verifyGbmvResult(const TestCaseConfig& tc,
                             const std::vector<float>& outputHost,
                             const std::vector<float>& goldenOutput) {
    const int64_t stride = tc.incy.has_value() ? tc.incy.value() : 1;
    const int64_t yCount = computeLogicalOutputCountGbmv(tc);
    const size_t verifyCount = (yCount > 0) ? static_cast<size_t>(yCount) : outputHost.size();
    return verifyStridedVector(tc, outputHost, goldenOutput, verifyCount, stride);
}
