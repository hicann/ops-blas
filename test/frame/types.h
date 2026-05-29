/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <optional>
#include <string>

enum class PrecisionMode {
    ABS,
    REL,
    COMBINED,
    MERE_MARE,
    EXACT,
    INTEGER
};

struct VerifyConfig {
    PrecisionMode mode = PrecisionMode::ABS;
    double absTol = 1e-4;
    double relTol = 1e-5;
    double mereThreshold = 1.0 / 8192.0;
    double mareMultiplier = 10.0;
    double epsilonForRel = 1e-7;
};

struct TestCaseConfig {
    std::string caseId;
    std::string level = "L0";
    std::string description;
    std::string dataType = "fp32";

    std::optional<std::string> trans;
    std::optional<int64_t> m;
    std::optional<int64_t> n;
    std::optional<int64_t> k;
    std::optional<int64_t> kl;
    std::optional<int64_t> ku;
    std::optional<int64_t> lda;
    std::optional<int64_t> ldb;
    std::optional<int64_t> ldc;
    std::optional<float> alphaReal;
    std::optional<float> alphaImag;
    std::optional<float> betaReal;
    std::optional<float> betaImag;
    std::optional<int64_t> incx;
    std::optional<int64_t> incy;
    std::optional<std::string> uplo;
    std::optional<std::string> side;
    std::optional<std::string> diag;
    std::optional<int64_t> batchCount;
    std::optional<uint32_t> seed;
    std::optional<int64_t> totalLength;
    std::optional<float> valueX;
    std::optional<float> sparam0;
    std::optional<float> sparam1;
    std::optional<float> sparam2;
    std::optional<float> sparam3;
    std::optional<float> sparam4;

    bool expectSuccess = true;

    VerifyConfig verifyCfg;

    std::optional<float> absTol;
    std::optional<float> relTol;
    std::optional<float> mereThreshold;
    std::optional<float> mareMultiplier;
};

struct OpConfig {
    std::string opName;
    std::string apiFunction;
    std::string inputBuffers = "x";
    std::string outputBuffers = "y";
    std::string resultType = "vector";
    std::string dataFormat = "general";
    std::string dataType = "fp32";
    std::string goldenFunction;
    std::string defaultPrecisionMode = "ABS";
};

#endif // TYPES_H
