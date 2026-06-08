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
    INDEX,       // 1, 2, 3, ...  (default)
    RANDOM,      // random uniform [-2, 2]
    ZEROS,       // all zeros
    ONES,        // all ones
    NULLPTR,     // null pointer (for error-path testing)
    SENTINEL,    // sentinel value (-999)
    MIXED,       // alternating positive/negative: (-1)^i * (i+1) * scale
    IDENTITY,    // identity matrix (1s on diagonal, 0 elsewhere)
    DIAGONAL,    // diagonal matrix with random diagonal values
    UPPER_TRI,   // upper triangular (random in upper triangle, 0 below)
    LOWER_TRI,   // lower triangular (random in lower triangle, 0 above)
    LARGE_RANGE, // random uniform [-1e6, 1e6]
    SMALL_RANGE, // random uniform [-1e-6, 1e-6]
    ILL_COND     // ill-conditioned matrix (columns with vastly different norms)
};

inline BlasDataFill parseDataFill(const std::string& s)
{
    static const std::pair<const char*, BlasDataFill> kMap[] = {
        {"INDEX", BlasDataFill::INDEX},
        {"index", BlasDataFill::INDEX},
        {"RANDOM", BlasDataFill::RANDOM},
        {"random", BlasDataFill::RANDOM},
        {"ZEROS", BlasDataFill::ZEROS},
        {"zeros", BlasDataFill::ZEROS},
        {"ONES", BlasDataFill::ONES},
        {"ones", BlasDataFill::ONES},
        {"NULLPTR", BlasDataFill::NULLPTR},
        {"nullptr", BlasDataFill::NULLPTR},
        {"SENTINEL", BlasDataFill::SENTINEL},
        {"sentinel", BlasDataFill::SENTINEL},
        {"MIXED", BlasDataFill::MIXED},
        {"mixed", BlasDataFill::MIXED},
        {"IDENTITY", BlasDataFill::IDENTITY},
        {"identity", BlasDataFill::IDENTITY},
        {"DIAGONAL", BlasDataFill::DIAGONAL},
        {"diagonal", BlasDataFill::DIAGONAL},
        {"UPPER_TRI", BlasDataFill::UPPER_TRI},
        {"upper_tri", BlasDataFill::UPPER_TRI},
        {"upper", BlasDataFill::UPPER_TRI},
        {"LOWER_TRI", BlasDataFill::LOWER_TRI},
        {"lower_tri", BlasDataFill::LOWER_TRI},
        {"lower", BlasDataFill::LOWER_TRI},
        {"LARGE_RANGE", BlasDataFill::LARGE_RANGE},
        {"large_range", BlasDataFill::LARGE_RANGE},
        {"SMALL_RANGE", BlasDataFill::SMALL_RANGE},
        {"small_range", BlasDataFill::SMALL_RANGE},
        {"ILL_COND", BlasDataFill::ILL_COND},
        {"ill_cond", BlasDataFill::ILL_COND},
    };
    for (const auto& entry : kMap) {
        if (s == entry.first)
            return entry.second;
    }
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

inline void FillStructuredMatrix(std::vector<float>& data, int m, int n, int lda, BlasDataFill fill, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    int mn = std::min(m, n);
    switch (fill) {
        case BlasDataFill::IDENTITY:
            for (int j = 0; j < mn; j++)
                data[j + static_cast<size_t>(j) * lda] = 1.0f;
            break;
        case BlasDataFill::DIAGONAL:
            for (int j = 0; j < mn; j++)
                data[j + static_cast<size_t>(j) * lda] = dist(rng);
            break;
        case BlasDataFill::UPPER_TRI:
            for (int j = 0; j < n; j++)
                for (int i = 0; i <= std::min(j, m - 1); i++)
                    data[i + static_cast<size_t>(j) * lda] = dist(rng);
            break;
        case BlasDataFill::LOWER_TRI:
            for (int j = 0; j < mn; j++)
                for (int i = j; i < m; i++)
                    data[i + static_cast<size_t>(j) * lda] = dist(rng);
            break;
        case BlasDataFill::ILL_COND:
            for (int j = 0; j < n; j++) {
                float scale = (j % 2 == 0) ? 1e6f : 1e-6f;
                for (int i = 0; i < m; i++)
                    data[i + static_cast<size_t>(j) * lda] = dist(rng) * scale;
            }
            break;
        default:
            break;
    }
}

// Full m×n column-major matrix with leading dimension lda.
inline std::vector<float> makeBlasMatrix(
    int m, int n, int lda, BlasDataFill fill, const std::string& desc = "", uint32_t seed = 0)
{
    if (fill == BlasDataFill::NULLPTR || m <= 0 || n <= 0 || lda <= 0)
        return {};

    const size_t storageSize = static_cast<size_t>(lda) * n;
    const size_t maxIndex = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * lda + 1;
    std::vector<float> data(std::max(storageSize, maxIndex), 0.0f);

    if (fill == BlasDataFill::ZEROS) {
        return data;
    }
    if (fill == BlasDataFill::ONES) {
        std::fill(data.begin(), data.end(), 1.0f);
        return data;
    }

    std::mt19937 rng(seed ? seed : 42);

    if (fill == BlasDataFill::IDENTITY || fill == BlasDataFill::DIAGONAL || fill == BlasDataFill::UPPER_TRI ||
        fill == BlasDataFill::LOWER_TRI || fill == BlasDataFill::ILL_COND) {
        FillStructuredMatrix(data, m, n, lda, fill, rng);
        return data;
    }

    // RANDOM / LARGE_RANGE / SMALL_RANGE / default
    float lo = -2.0f;
    float hi = 2.0f;
    if (fill == BlasDataFill::LARGE_RANGE) {
        lo = -1e6f;
        hi = 1e6f;
    } else if (fill == BlasDataFill::SMALL_RANGE) {
        lo = -1e-6f;
        hi = 1e-6f;
    }
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t idx = 0; idx < storageSize; idx++)
        data[idx] = dist(rng);
    return data;
}

#endif // FILL_H
