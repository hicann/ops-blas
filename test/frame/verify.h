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

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "types.h"

class Verifier {
public:
    static bool verifyVector(const float* output, const float* golden,
                              size_t count, int64_t stride,
                              const VerifyConfig& cfg, const std::string& caseId) {
        switch (cfg.mode) {
            case PrecisionMode::ABS:       return verifyAbs(output, golden, count, stride, cfg.absTol, caseId);
            case PrecisionMode::REL:       return verifyRel(output, golden, count, stride, cfg.relTol, cfg.epsilonForRel, caseId);
            case PrecisionMode::COMBINED:  return verifyCombined(output, golden, count, stride, cfg.absTol, cfg.relTol, caseId);
            case PrecisionMode::MERE_MARE: return verifyMereMare(output, golden, count, stride, cfg.mereThreshold, cfg.mareMultiplier, caseId);
            case PrecisionMode::EXACT:     return verifyExact(output, golden, count, stride, caseId);
            case PrecisionMode::INTEGER:   return verifyIntegerVec(output, golden, count, stride, caseId);
            default:                       return verifyAbs(output, golden, count, stride, cfg.absTol, caseId);
        }
    }

    static bool verifyScalar(float output, float golden,
                              const VerifyConfig& cfg, const std::string& caseId) {
        std::cout << "[" << caseId << "] Output: " << output << std::endl;
        std::cout << "[" << caseId << "] Golden: " << golden << std::endl;

        bool pass = false;
        switch (cfg.mode) {
            case PrecisionMode::ABS:
                pass = std::abs(output - golden) < cfg.absTol;
                break;
            case PrecisionMode::REL:
                pass = std::abs(output - golden) / (std::abs(golden) + cfg.epsilonForRel) < cfg.relTol;
                break;
            default:
                pass = std::abs(output - golden) < cfg.absTol;
                break;
        }

        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (diff=" << std::abs(output - golden) << ")" << std::endl;
        return pass;
    }

    static bool verifyInteger(int64_t output, int64_t golden, const std::string& caseId) {
        std::cout << "[" << caseId << "] Output: " << output << std::endl;
        std::cout << "[" << caseId << "] Golden: " << golden << std::endl;
        bool pass = (output == golden);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << std::endl;
        return pass;
    }

private:
    static void printHead(const float* data, size_t count, int64_t stride,
                          const std::string& label, const std::string& caseId) {
        std::cout << std::fixed << std::setprecision(6);
        constexpr size_t kMaxPrint = 10;
        std::cout << "[" << caseId << "] " << label << ": ";
        for (size_t i = 0; i < count && i < kMaxPrint; i++) {
            std::cout << data[i * static_cast<size_t>(stride)] << " ";
        }
        if (count > kMaxPrint) std::cout << "...";
        std::cout << std::endl;
    }

    static bool verifyAbs(const float* output, const float* golden,
                          size_t count, int64_t stride, double absTol,
                          const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        size_t failCount = 0;
        for (size_t i = 0; i < count; i++) {
            float diff = std::abs(output[i * static_cast<size_t>(stride)] - golden[i * static_cast<size_t>(absStride)]);
            if (diff > absTol) failCount++;
        }

        bool pass = (failCount == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (absTol=" << absTol << ", " << failCount << "/" << count << " failures)" << std::endl;
        return pass;
    }

    static bool verifyRel(const float* output, const float* golden,
                          size_t count, int64_t stride,
                          double relTol, double eps,
                          const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        double maxRelErr = 0.0;
        for (size_t i = 0; i < count; i++) {
            float outVal = output[i * static_cast<size_t>(stride)];
            float goldVal = golden[i * static_cast<size_t>(absStride)];
            double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + eps);
            if (relErr > maxRelErr) maxRelErr = relErr;
        }

        bool pass = (maxRelErr < relTol);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (maxRelErr=" << maxRelErr << ", relTol=" << relTol << ")" << std::endl;
        return pass;
    }

    static bool verifyCombined(const float* output, const float* golden,
                                size_t count, int64_t stride,
                                double absTol, double relTol,
                                const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        size_t failCount = 0;
        for (size_t i = 0; i < count; i++) {
            float outVal = output[i * static_cast<size_t>(stride)];
            float goldVal = golden[i * static_cast<size_t>(absStride)];
            double diff = std::abs(outVal - goldVal);
            double scale = std::abs(goldVal) + 1e-7;
            if (diff > absTol && diff > relTol * scale) failCount++;
        }

        bool pass = (failCount == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (absTol=" << absTol << ", relTol=" << relTol << ", "
                  << failCount << "/" << count << " failures)" << std::endl;
        return pass;
    }

    static bool verifyMereMare(const float* output, const float* golden,
                                size_t count, int64_t stride,
                                double threshold, double multiplier,
                                const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        constexpr double kEpsilon = 1e-7;
        double outlierLimit = multiplier * threshold;
        double sumRelErr = 0.0;
        double maxRelErr = 0.0;
        size_t outlierCount = 0;

        for (size_t i = 0; i < count; i++) {
            float outVal = output[i * static_cast<size_t>(stride)];
            float goldVal = golden[i * static_cast<size_t>(absStride)];
            double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + kEpsilon);
            sumRelErr += relErr;
            if (relErr > maxRelErr) maxRelErr = relErr;
            if (relErr > outlierLimit) outlierCount++;
        }

        double mere = (count > 0) ? sumRelErr / static_cast<double>(count) : 0.0;

        std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr
                  << " (threshold=" << threshold << ", outlier_limit=" << outlierLimit << ")" << std::endl;

        bool pass = (mere < threshold) && (maxRelErr < outlierLimit);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (MERE < threshold && MARE < " << multiplier << "*threshold, "
                  << outlierCount << " outliers out of " << count << " elements)" << std::endl;
        return pass;
    }

    static bool verifyExact(const float* output, const float* golden,
                             size_t count, int64_t stride,
                             const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        size_t failCount = 0;
        for (size_t i = 0; i < count; i++) {
            const float outVal = output[i * static_cast<size_t>(stride)];
            const float goldVal = golden[i * static_cast<size_t>(absStride)];
            if (std::isnan(outVal) && std::isnan(goldVal)) {
                continue;
            }
            if (outVal != goldVal) {
                failCount++;
            }
        }

        bool pass = (failCount == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (exact match, " << failCount << "/" << count << " mismatches)" << std::endl;
        return pass;
    }

    static bool verifyIntegerVec(const float* output, const float* golden,
                                  size_t count, int64_t stride,
                                  const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        int64_t absStride = (stride < 0) ? -stride : stride;
        printHead(golden, count, absStride, "Golden", caseId);

        size_t failCount = 0;
        for (size_t i = 0; i < count; i++) {
            int64_t outVal = static_cast<int64_t>(output[i * static_cast<size_t>(stride)]);
            int64_t goldVal = static_cast<int64_t>(golden[i * static_cast<size_t>(absStride)]);
            if (outVal != goldVal) failCount++;
        }

        bool pass = (failCount == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (integer match, " << failCount << "/" << count << " mismatches)" << std::endl;
        return pass;
    }
};

inline bool verifyDenseVector(const TestCaseConfig& tc,
                              const std::vector<float>& output,
                              const std::vector<float>& golden) {
    const size_t count = output.size();
    return Verifier::verifyVector(output.data(), golden.data(), count,
                                  1, tc.verifyCfg, tc.caseId);
}

inline bool verifyStridedVector(const TestCaseConfig& tc,
                                const std::vector<float>& outputHost,
                                const std::vector<float>& goldenOutput,
                                size_t count, int64_t stride) {
    const int64_t absStride = (stride < 0) ? -stride : stride;
    const float* outputPtr = outputHost.data();
    if (stride < 0 && count > 0) {
        outputPtr += static_cast<int64_t>(count - 1) * absStride;
    }
    return Verifier::verifyVector(outputPtr, goldenOutput.data(), count,
                                  stride, tc.verifyCfg, tc.caseId);
}
