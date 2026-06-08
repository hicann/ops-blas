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
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

constexpr float kBlasSentinel = -999.0f;

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1: Configuration
//
// 命名规则: METHOD_PATTERN_VAL...
//   METHOD:  NULLPTR | INDEX | RANDOM | VALUE
//   PATTERN: NORM | UPPER | LOWER | DIAG | ALTER | EXTREME | ILLCOND | BANDED
//   VAL:     数值(N前缀=负, P可省略) 或特殊标记
//   参数化 PATTERN (如 BANDED) 先消耗结构参数, 剩余 VAL 用于填充值
//   从某位开始可不填(后续取默认值), 不允许跳位
// ─────────────────────────────────────────────────────────────────────────────

struct BlasFillMode {
    enum Method { M_NULLPTR, M_INDEX, M_RANDOM, M_VALUE } method = M_INDEX;
    enum Pattern { P_NORM, P_UPPER, P_LOWER, P_DIAG, P_ALTER, P_EXTREME, P_ILLCOND, P_BANDED } pattern = P_NORM;
    float val1 = 1.0f;
    float val2 = 1.0f;
    int bandedKl = 0;
    int bandedKu = 0;

    BlasFillMode() = default;
    BlasFillMode(const std::string& s) { parse(s); }

private:
    static const std::map<std::string, Method>& methodMap()
    {
        static const std::map<std::string, Method> m = {
            {"NULLPTR", M_NULLPTR}, {"INDEX", M_INDEX}, {"RANDOM", M_RANDOM}, {"VALUE", M_VALUE}};
        return m;
    }

    static const std::map<std::string, Pattern>& patternMap()
    {
        static const std::map<std::string, Pattern> m = {
            {"NORM", P_NORM},   {"UPPER", P_UPPER},     {"LOWER", P_LOWER},     {"DIAG", P_DIAG},
            {"ALTER", P_ALTER}, {"EXTREME", P_EXTREME}, {"ILLCOND", P_ILLCOND}, {"BANDED", P_BANDED}};
        return m;
    }

    static int patternStructParams(Pattern p)
    {
        static const std::map<Pattern, int> m = {{P_BANDED, 2}};
        auto it = m.find(p);
        return (it != m.end()) ? it->second : 0;
    }

    static const std::map<std::string, float>& specialValues()
    {
        static const std::map<std::string, float> m = {
            {"INF", INFINITY}, {"NAN", NAN}, {"ALTER", 0.0f}, {"EXTREME", 0.0f}, {"ILLCOND", 0.0f}};
        return m;
    }

    static float parseNum(const std::string& s)
    {
        auto it = specialValues().find(s);
        if (it != specialValues().end())
            return it->second;

        bool neg = false;
        size_t pos = 0;
        if (pos < s.size() && s[pos] == 'N') {
            neg = true;
            pos++;
        } else if (pos < s.size() && s[pos] == 'P') {
            pos++;
        }

        std::string numStr = s.substr(pos);
        float val = 0.0f;
        if (!numStr.empty()) {
            size_t epos = numStr.find('E');
            if (epos != std::string::npos) {
                std::string mantissa = numStr.substr(0, epos);
                std::string expPart = numStr.substr(epos + 1);
                float mant = mantissa.empty() ? 1.0f : std::stof(mantissa);
                bool expNeg = false;
                size_t eidx = 0;
                if (eidx < expPart.size() && expPart[eidx] == 'N') {
                    expNeg = true;
                    eidx++;
                }
                std::string expNum = expPart.substr(eidx);
                float exp = expNum.empty() ? 0.0f : std::stof(expNum);
                if (expNeg)
                    exp = -exp;
                val = mant * std::pow(10.0f, exp);
            } else {
                val = std::stof(numStr);
            }
        }
        return neg ? -val : val;
    }

    void parse(const std::string& s)
    {
        std::vector<std::string> parts;
        std::istringstream iss(s);
        std::string token;
        while (std::getline(iss, token, '_'))
            if (!token.empty())
                parts.push_back(token);

        if (parts.empty())
            return;

        auto mIt = methodMap().find(parts[0]);
        if (mIt == methodMap().end())
            return;
        method = mIt->second;

        if (method == M_NULLPTR)
            return;

        static const std::map<Method, std::pair<float, float>> defaults = {
            {M_INDEX, {1.0f, 1.0f}}, {M_RANDOM, {-1.0f, -1.0f}}, {M_VALUE, {0.0f, 0.0f}}};
        auto dIt = defaults.find(method);
        if (dIt != defaults.end()) {
            val1 = dIt->second.first;
            val2 = dIt->second.second;
        }

        size_t idx = 1;
        if (idx < parts.size()) {
            auto pIt = patternMap().find(parts[idx]);
            if (pIt != patternMap().end()) {
                pattern = pIt->second;
                idx++;
            }
        }

        int structParams = patternStructParams(pattern);
        if (structParams >= 1 && idx < parts.size()) {
            bandedKl = static_cast<int>(parseNum(parts[idx]));
            idx++;
        }
        if (structParams >= 2 && idx < parts.size()) {
            bandedKu = static_cast<int>(parseNum(parts[idx]));
            idx++;
        }

        if (idx < parts.size()) {
            val1 = parseNum(parts[idx]);
            val2 = val1;
            idx++;
        }
        if (idx < parts.size())
            val2 = parseNum(parts[idx]);
    }
};

inline BlasFillMode parseFill(const std::string& s) { return BlasFillMode(s); }

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2: Strategy — 值生成策略
// ─────────────────────────────────────────────────────────────────────────────

class ValueGenerator {
public:
    virtual ~ValueGenerator() = default;
    virtual float at(size_t index) = 0;
};

class SequenceGenerator : public ValueGenerator {
    float start_;

public:
    explicit SequenceGenerator(float start) : start_(start) {}
    float at(size_t i) override { return (start_ < 0.0f ? -1.0f : 1.0f) * (std::abs(start_) + static_cast<float>(i)); }
};

class ConstantGenerator : public ValueGenerator {
    float value_;

public:
    explicit ConstantGenerator(float v) : value_(v) {}
    float at(size_t) override { return value_; }
};

class RandomGenerator : public ValueGenerator {
    std::mt19937& rng_;
    std::uniform_real_distribution<float> dist_;

public:
    RandomGenerator(std::mt19937& rng, float lo, float hi) : rng_(rng), dist_(lo, hi) {}
    float at(size_t) override { return dist_(rng_); }
};

class AlterGenerator : public ValueGenerator {
public:
    float at(size_t i) override { return (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1); }
};

class ExtremeGenerator : public ValueGenerator {
    static constexpr float kVals[] = {1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, FLT_TRUE_MIN};

public:
    float at(size_t i) override { return kVals[i % 7]; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2.5: Factory — map 驱动的策略创建
// ─────────────────────────────────────────────────────────────────────────────

inline std::unique_ptr<ValueGenerator> createGenerator(const BlasFillMode& mode, std::mt19937& rng)
{
    static const std::map<BlasFillMode::Pattern, std::function<std::unique_ptr<ValueGenerator>()>> patternGens = {
        {BlasFillMode::P_ALTER, []() { return std::make_unique<AlterGenerator>(); }},
        {BlasFillMode::P_EXTREME, []() { return std::make_unique<ExtremeGenerator>(); }},
    };

    auto pIt = patternGens.find(mode.pattern);
    if (pIt != patternGens.end())
        return pIt->second();

    static const std::map<
        BlasFillMode::Method, std::function<std::unique_ptr<ValueGenerator>(const BlasFillMode&, std::mt19937&)>>
        methodGens = {
            {BlasFillMode::M_INDEX,
             [](const BlasFillMode& m, std::mt19937&) { return std::make_unique<SequenceGenerator>(m.val1); }},
            {BlasFillMode::M_VALUE,
             [](const BlasFillMode& m, std::mt19937&) { return std::make_unique<ConstantGenerator>(m.val1); }},
            {BlasFillMode::M_RANDOM,
             [](const BlasFillMode& m, std::mt19937& r) {
                 float lo = (m.val1 < 0.0f) ? -FLT_MAX : -m.val1;
                 float hi = (m.val2 < 0.0f) ? FLT_MAX : m.val2;
                 return std::make_unique<RandomGenerator>(r, lo, hi);
             }},
        };

    auto mIt = methodGens.find(mode.method);
    if (mIt != methodGens.end())
        return mIt->second(mode, rng);

    return std::make_unique<SequenceGenerator>(1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3: Layout — 内存布局函数
//   核心版本接受 BlasFillMode, string 版本构造后委托
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<float> makeBlasArray(int64_t size, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0)
        return {};

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    std::vector<float> data(static_cast<size_t>(size));
    for (size_t i = 0; i < data.size(); i++)
        data[i] = gen->at(i);
    return data;
}

inline std::vector<float> makeBlasArray(int64_t size, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasArray(size, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const size_t aSize = (n > 0) ? static_cast<size_t>(lda) * n : 1;
    std::vector<float> data(aSize, 0.0f);
    if (n <= 0)
        return data;

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    size_t seq = 0;
    for (int j = 0; j < n; j++) {
        int iStart = std::max(0, j - ku);
        int iEnd = std::min(m - 1, j + kl);
        for (int i = iStart; i <= iEnd; i++) {
            int bandRow = ku + i - j;
            data[bandRow + j * lda] = gen->at(seq++);
        }
    }
    return data;
}

inline std::vector<float> makeBlasBanded(
    int m, int n, int kl, int ku, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasBanded(m, n, kl, ku, lda, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasBanded(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    return makeBlasBanded(m, n, fill.bandedKl, fill.bandedKu, lda, fill, seed);
}

inline std::vector<float> makeBlasBanded(int m, int n, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    BlasFillMode fill(fillStr);
    return makeBlasBanded(m, n, fill.bandedKl, fill.bandedKu, lda, fill, seed);
}

inline std::vector<float> makeBlasStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const int absInc = std::abs(inc);
    const size_t size = (count > 0) ? static_cast<size_t>((count - 1) * absInc + 1) : 1;
    std::vector<float> data(size, 0.0f);
    if (count <= 0)
        return data;

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    for (int i = 0; i < count; i++) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = gen->at(i);
    }
    return data;
}

inline std::vector<float> makeBlasStrided(int count, int inc, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasStrided(count, inc, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasPacked(int n, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
    std::vector<float> data(apLen);

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    bool upper = (fill.pattern == BlasFillMode::P_UPPER);
    size_t idx = 0;
    for (int j = 0; j < n; j++) {
        int iStart = upper ? 0 : j;
        int iEnd = upper ? j : (n - 1);
        for (int i = iStart; i <= iEnd; i++) {
            data[idx] = gen->at(idx);
            idx++;
        }
    }
    return data;
}

inline std::vector<float> makeBlasPacked(int n, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasPacked(n, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasTriangular(int n, bool upper, const BlasFillMode& fill, uint32_t seed = 0)
{
    BlasFillMode adjusted = fill;
    adjusted.pattern = upper ? BlasFillMode::P_UPPER : BlasFillMode::P_LOWER;
    return makeBlasPacked(n, adjusted, seed);
}

inline std::vector<float> makeBlasTriangular(int n, bool upper, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasTriangular(n, upper, BlasFillMode(fillStr), seed);
}

inline void FillStructuredMatrix(
    std::vector<float>& data, int m, int n, int lda, BlasFillMode::Pattern pattern, std::mt19937& rng)
{
    static const std::map<BlasFillMode::Pattern, std::function<void(std::vector<float>&, int, int, int, std::mt19937&)>>
        layoutMap = {
            {BlasFillMode::P_UPPER,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 for (int j = 0; j < n; j++)
                     for (int i = 0; i <= std::min(j, m - 1); i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_LOWER,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 int mn = std::min(m, n);
                 for (int j = 0; j < mn; j++)
                     for (int i = j; i < m; i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_DIAG,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 int mn = std::min(m, n);
                 for (int j = 0; j < mn; j++)
                     d[j + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_ILLCOND,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 for (int j = 0; j < n; j++) {
                     float scale = (j % 2 == 0) ? 1e6f : 1e-6f;
                     for (int i = 0; i < m; i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r) * scale;
                 }
             }},
        };

    auto it = layoutMap.find(pattern);
    if (it != layoutMap.end())
        it->second(data, m, n, lda, rng);
}

inline std::vector<float> makeBlasMatrix(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0)
        return {};

    const size_t storageSize = static_cast<size_t>(lda) * n;
    const size_t maxIndex = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * lda + 1;
    std::vector<float> data(std::max(storageSize, maxIndex), 0.0f);

    if (fill.method == BlasFillMode::M_VALUE) {
        if (fill.pattern == BlasFillMode::P_DIAG) {
            int mn = std::min(m, n);
            for (int j = 0; j < mn; j++)
                data[j + static_cast<size_t>(j) * lda] = fill.val1;
            return data;
        }
        std::fill(data.begin(), data.end(), fill.val1);
        return data;
    }

    static const std::map<BlasFillMode::Pattern, bool> structured = {
        {BlasFillMode::P_UPPER, true},
        {BlasFillMode::P_LOWER, true},
        {BlasFillMode::P_DIAG, true},
        {BlasFillMode::P_ILLCOND, true}};
    if (structured.count(fill.pattern)) {
        std::mt19937 rng(seed ? seed : 42);
        FillStructuredMatrix(data, m, n, lda, fill.pattern, rng);
        return data;
    }

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);
    for (size_t i = 0; i < storageSize; i++)
        data[i] = gen->at(i);
    return data;
}

inline std::vector<float> makeBlasMatrix(int m, int n, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasMatrix(m, n, lda, BlasFillMode(fillStr), seed);
}

#endif // FILL_H
