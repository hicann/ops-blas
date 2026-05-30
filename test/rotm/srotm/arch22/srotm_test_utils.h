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
#include <cstdint>
#include <random>
#include <vector>

#include "data.h"
#include "types.h"
#include "verify.h"

inline size_t computeSrotmStorageSize(int64_t n, int64_t inc)
{
    if (n <= 0) {
        return 1;
    }
    const int64_t absInc = (inc < 0) ? -inc : inc;
    return static_cast<size_t>((n - 1) * absInc + 1);
}

inline std::vector<std::vector<float>> generateHostDataSrotm(const TestCaseConfig& tc)
{
    const uint32_t seed = tc.seed.value_or(20260411);
    const int64_t n = tc.n.value_or(0);
    const int64_t incx = tc.incx.value_or(1);
    const int64_t incy = tc.incy.value_or(1);

    std::vector<float> x(computeSrotmStorageSize(n, incx), -999.0f);
    std::vector<float> y(computeSrotmStorageSize(n, incy), -777.0f);

    if (n > 0) {
        std::mt19937 rng(seed);
        DataGenerator::fillStridedVector(x.data(), n, incx, rng);
        DataGenerator::fillStridedVector(y.data(), n, incy, rng);
    }

    return {std::move(x), std::move(y)};
}

inline bool verifySrotmResult(const TestCaseConfig& tc,
                              const std::vector<float>& outputHost,
                              const std::vector<float>& goldenOutput)
{
    const size_t count = std::min(outputHost.size(), goldenOutput.size());
    return Verifier::verifyVector(
        outputHost.data(), goldenOutput.data(), count, 1, tc.verifyCfg, tc.caseId);
}
