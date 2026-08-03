/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VERIFY_UPLO_H
#define VERIFY_UPLO_H

#include <string>
#include <vector>
#include "verify.h"

template <typename Param>
static inline void VerifyUploTriangle(
    const Param& p, const float* cPtr, const float* goldPtr, size_t cSize)
{
    if (cSize == 0) {
        return;
    }

    std::vector<float> npuUplo;
    std::vector<float> goldenUplo;
    std::vector<float> npuNonUplo;
    std::vector<float> goldenNonUplo;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUplo.push_back(cPtr[idx]);
                goldenUplo.push_back(goldPtr[idx]);
            } else {
                npuNonUplo.push_back(cPtr[idx]);
                goldenNonUplo.push_back(goldPtr[idx]);
            }
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenUplo.data(), goldenUplo.size());
    EXPECT_TRUE(Verifier::verifyVector(npuUplo.data(), goldenUplo.data(), npuUplo.size(), 1, cfg, p.caseName));

    if (!npuNonUplo.empty()) {
        VerifyConfig cfgNonUplo;
        applyMixedTolerance(cfgNonUplo, ACL_FLOAT, goldenNonUplo.data(), goldenNonUplo.size());
        EXPECT_TRUE(Verifier::verifyVector(npuNonUplo.data(), goldenNonUplo.data(),
            npuNonUplo.size(), 1, cfgNonUplo, std::string(p.caseName) + "_nonuplo"));
    }
}

#endif // VERIFY_UPLO_H
