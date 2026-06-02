/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FILL_H
#define FILL_H

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// ── BlasDataFill: how to initialise an array/buffer ──
enum class BlasDataFill {
    INDEX,    // 1, 2, 3, ...  (default)
    RANDOM,   // random uniform [-2, 2]
    ZEROS,    // all zeros
    ONES,     // all ones
    NULLPTR,  // null pointer (for error-path testing)
    SENTINEL, // sentinel fill (-999)
    MIXED     // alternating positive/negative: (-1)^i * (i+1) * scale
};

inline BlasDataFill parseDataFill(const std::string& s)
{
    if (s == "INDEX" || s == "index")
        return BlasDataFill::INDEX;
    if (s == "RANDOM" || s == "random")
        return BlasDataFill::RANDOM;
    if (s == "ZEROS" || s == "zeros")
        return BlasDataFill::ZEROS;
    if (s == "ONES" || s == "ones")
        return BlasDataFill::ONES;
    if (s == "NULLPTR" || s == "nullptr")
        return BlasDataFill::NULLPTR;
    if (s == "SENTINEL" || s == "sentinel")
        return BlasDataFill::SENTINEL;
    if (s == "MIXED" || s == "mixed")
        return BlasDataFill::MIXED;
    return BlasDataFill::INDEX;
}

constexpr float kBlasSentinel = -999.0f;

inline std::vector<float> makeBlasArray(
    int64_t size, BlasDataFill fill, const std::string& desc = "", float sentinel = kBlasSentinel, uint32_t seed = 0)
{
    if (fill == BlasDataFill::NULLPTR || size <= 0)
        return {};

    std::vector<float> data(static_cast<size_t>(size));

    if (desc.find("large") != std::string::npos) {
        std::fill(data.begin(), data.end(), 1e10f);
    } else if (desc.find("neg") != std::string::npos) {
        for (size_t i = 0; i < data.size(); i++)
            data[i] = -static_cast<float>(i + 1);
    } else if (desc.find("inf") != std::string::npos) {
        std::fill(data.begin(), data.end(), INFINITY);
    } else if (desc.find("nan") != std::string::npos) {
        std::fill(data.begin(), data.end(), NAN);
    } else if (desc.find("extr") != std::string::npos) {
        const float vals[] = {1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
        for (size_t i = 0; i < data.size(); i++)
            data[i] = vals[i % 7];
    } else {
        switch (fill) {
            case BlasDataFill::ZEROS:
                std::fill(data.begin(), data.end(), 0.0f);
                break;
            case BlasDataFill::ONES:
                std::fill(data.begin(), data.end(), 1.0f);
                break;
            case BlasDataFill::SENTINEL:
                std::fill(data.begin(), data.end(), sentinel);
                break;
            case BlasDataFill::RANDOM: {
                std::mt19937 rng(seed ? seed : 42);
                std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                for (auto& v : data)
                    v = dist(rng);
                break;
            }
            case BlasDataFill::INDEX:
            default:
                for (size_t i = 0; i < data.size(); i++)
                    data[i] = static_cast<float>(i + 1);
                break;
        }
    }

    return data;
}

// banded matrix — fills the band region defined by kl, ku
inline std::vector<float> makeBlasBanded(int m, int n, int kl, int ku, int lda, BlasDataFill fill, uint32_t seed = 0)
{
    if (fill == BlasDataFill::NULLPTR)
        return {};

    const size_t aSize = (n > 0) ? static_cast<size_t>(lda) * n : 1;
    std::vector<float> data(aSize, 0.0f);
    if (n <= 0)
        return data;

    std::mt19937 rng(seed ? seed : 42);
    auto randVal = [&]() {
        if (fill == BlasDataFill::INDEX) {
            static size_t idx = 1;
            return static_cast<float>(idx++);
        }
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        return dist(rng);
    };

    size_t seq = 1;
    for (int j = 0; j < n; j++) {
        int iStart = std::max(0, j - ku);
        int iEnd = std::min(m - 1, j + kl);
        for (int i = iStart; i <= iEnd; i++) {
            int bandRow = ku + i - j;
            data[bandRow + j * lda] = (fill == BlasDataFill::INDEX) ? static_cast<float>(seq++) : randVal();
        }
    }
    return data;
}

// strided vector — count elements spaced by inc
inline std::vector<float> makeBlasStrided(int count, int inc, BlasDataFill fill, uint32_t seed = 0)
{
    if (fill == BlasDataFill::NULLPTR)
        return {};

    const int absInc = std::abs(inc);
    const size_t size = (count > 0) ? static_cast<size_t>((count - 1) * absInc + 1) : 1;
    std::vector<float> data(size, 0.0f);
    if (count <= 0)
        return data;

    std::mt19937 rng(seed ? seed : 42);
    for (int i = 0; i < count; i++) {
        float v = 0.0f;
        switch (fill) {
            case BlasDataFill::ZEROS:
                v = 0.0f;
                break;
            case BlasDataFill::ONES:
                v = 1.0f;
                break;
            case BlasDataFill::INDEX:
                v = static_cast<float>(i + 1);
                break;
            case BlasDataFill::MIXED:
                v = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
                break;
            default: {
                std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                v = dist(rng);
                break;
            }
        }
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = v;
    }
    return data;
}

// triangular packed matrix — n*(n+1)/2 elements in column-major storage order
inline std::vector<float> makeBlasTriangular(
    int n, bool upper, BlasDataFill fill, const std::string& desc = "", uint32_t seed = 0)
{
    if (fill == BlasDataFill::NULLPTR)
        return {};

    const size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
    std::vector<float> data(apLen);

    if (desc.find("large") != std::string::npos) {
        std::fill(data.begin(), data.end(), 1e10f);
    } else if (desc.find("neg") != std::string::npos) {
        for (size_t i = 0; i < apLen; i++)
            data[i] = -static_cast<float>(i + 1);
    } else if (desc.find("inf") != std::string::npos) {
        std::fill(data.begin(), data.end(), INFINITY);
    } else if (desc.find("nan") != std::string::npos) {
        std::fill(data.begin(), data.end(), NAN);
    } else if (desc.find("extr") != std::string::npos) {
        const float vals[] = {1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};
        for (size_t i = 0; i < apLen; i++)
            data[i] = vals[i % 7];
    } else {
        std::mt19937 rng(seed ? seed : 42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        auto fillValue = [&](size_t idx) -> float {
            if (fill == BlasDataFill::ZEROS)
                return 0.0f;
            if (fill == BlasDataFill::ONES)
                return 1.0f;
            if (fill == BlasDataFill::RANDOM)
                return dist(rng);
            if (fill == BlasDataFill::MIXED)
                return (idx % 2 == 0) ? 0.1f * static_cast<float>(idx + 1) : -0.1f * static_cast<float>(idx + 1);
            return static_cast<float>(idx + 1);
        };
        size_t idx = 0;
        for (int j = 0; j < n; j++) {
            int iStart = upper ? 0 : j;
            int iEnd = upper ? j : (n - 1);
            for (int i = iStart; i <= iEnd; i++) {
                data[idx] = fillValue(idx);
                idx++;
            }
        }
    }

    return data;
}

#endif // FILL_H
