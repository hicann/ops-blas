/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for the details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEss FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <iomanip>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

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

constexpr float RTOL = 1e-4f;
constexpr float ATOL = 1e-5f;

static uint32_t g_seed = 0;

inline int32_t fastRand(void) {
    g_seed = g_seed * 1103515245 + 12345;
    return static_cast<int32_t>(g_seed / 65536) % 32768;
}

inline float frand(void) {
    return static_cast<float>(fastRand()) / 32768.0f;
}

void ComputeGoldenTrsvLower(aclblasDiagType diag, aclblasOperation trans, int64_t n,
                            const std::vector<float>& A, int64_t lda,
                            std::vector<float>& x)
{
    for (int64_t i = 0; i < n; i++) {
        float sum = x[i];
        for (int64_t j = 0; j < i; j++) {
            int64_t aOffset = (trans == ACLBLAS_OP_T) ? (j * lda + i) : (i * lda + j);
            sum -= A[aOffset] * x[j];
        }
        if (diag == ACLBLAS_NON_UNIT) {
            sum /= A[i * lda + i];
        }
        x[i] = sum;
    }
}

void ComputeGoldenTrsvUpper(aclblasDiagType diag, aclblasOperation trans, int64_t n,
                            const std::vector<float>& A, int64_t lda,
                            std::vector<float>& x)
{
    for (int64_t i = n - 1; i >= 0; i--) {
        float sum = x[i];
        for (int64_t j = i + 1; j < n; j++) {
            int64_t aOffset = (trans == ACLBLAS_OP_T) ? (j * lda + i) : (i * lda + j);
            sum -= A[aOffset] * x[j];
        }
        if (diag == ACLBLAS_NON_UNIT) {
            sum /= A[i * lda + i];
        }
        x[i] = sum;
    }
}

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden, int64_t n)
{
    bool allMatch = true;
    size_t maxErrors = 10;
    size_t errorCount = 0;

    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::fabs(output[i] - golden[i]);
        float relDiff = (golden[i] != 0.0f) ? diff / std::fabs(golden[i]) : diff;

        if (diff > ATOL && relDiff > RTOL) {
            if (errorCount < maxErrors) {
                std::cout << "Error at index " << i << ": output=" << output[i]
                          << ", golden=" << golden[i]
                          << ", diff=" << diff << std::endl;
            }
            errorCount++;
            allMatch = false;
        }
    }

    if (allMatch) {
        std::cout << "[Success] Case accuracy verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy verification failed! " << errorCount << " errors found." << std::endl;
        return 1;
    }
}

void PrintUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n <size>      Matrix dimension (default: 128)" << std::endl;
    std::cout << "  -s <seed>      Random seed (default: time-based)" << std::endl;
    std::cout << "  -r <range>     Random value range (default: 1.0)" << std::endl;
    std::cout << "  -d <diag>     Diagonal type: unit/nonunit (default: nonunit)" << std::endl;
    std::cout << "  -u <uplo>      Triangle: upper/lower (default: lower)" << std::endl;
    std::cout << "  -t <trans>     Transpose: n/t (default: n)" << std::endl;
    std::cout << "  -i <incx>      Increment for x (default: 1)" << std::endl;
    std::cout << "  -F             Fixed params mode (disable random params, default: random)" << std::endl;
    std::cout << "  -p <count>     Number of elements to preview (default: 10)" << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << progName << "              # Random params: uplo/trans/diag/n all random" << std::endl;
    std::cout << "  " << progName << " -s 42        # Reproducible random run" << std::endl;
    std::cout << "  " << progName << " -F -n 128 -u upper -t t  # Fixed params mode" << std::endl;
    std::cout << "  " << progName << " -p 20        # Show first 20 elements comparison" << std::endl;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    int64_t n = 128;
    float randomRange = 1.0f;
    bool randomParams = true;
    bool nSpecified = false, uploSpecified = false, transSpecified = false, diagSpecified = false, incxSpecified = false;
    int64_t previewCount = 10;
    aclblasDiagType diag = ACLBLAS_NON_UNIT;
    aclblasFillMode uplo = ACLBLAS_LOWER;
    aclblasOperation trans = ACLBLAS_OP_N;
    int64_t incx = 1;

    for (int32_t i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "-n" && i + 1 < argc) {
            n = std::atoi(argv[++i]);
            nSpecified = true;
        } else if (arg == "-s" && i + 1 < argc) {
            g_seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "-r" && i + 1 < argc) {
            randomRange = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "-d" && i + 1 < argc) {
            std::string diagStr = argv[++i];
            if (diagStr == "unit") {
                diag = ACLBLAS_UNIT;
            } else {
                diag = ACLBLAS_NON_UNIT;
            }
            diagSpecified = true;
        } else if (arg == "-u" && i + 1 < argc) {
            std::string uploStr = argv[++i];
            if (uploStr == "upper") {
                uplo = ACLBLAS_UPPER;
            } else {
                uplo = ACLBLAS_LOWER;
            }
            uploSpecified = true;
        } else if (arg == "-t" && i + 1 < argc) {
            std::string transStr = argv[++i];
            if (transStr == "t" || transStr == "T") {
                trans = ACLBLAS_OP_T;
            } else {
                trans = ACLBLAS_OP_N;
            }
            transSpecified = true;
        } else if (arg == "-i" && i + 1 < argc) {
            incx = std::atoi(argv[++i]);
            incxSpecified = true;
        } else if (arg == "-F" || arg == "--fixed-params") {
            randomParams = false;
            diag = ACLBLAS_NON_UNIT;
            uplo = ACLBLAS_LOWER;
            trans = ACLBLAS_OP_N;
            incx = 1;
            n = 128;
        } else if (arg == "-p" && i + 1 < argc) {
            previewCount = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (g_seed == 0) {
        g_seed = static_cast<uint32_t>(time(nullptr));
    }

    if (randomParams) {
        if (!uploSpecified) {
            uplo = (fastRand() % 2 == 0) ? ACLBLAS_LOWER : ACLBLAS_UPPER;
        }
        if (!transSpecified) {
            trans = (fastRand() % 2 == 0) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
        }
        if (!diagSpecified) {
            diag = (fastRand() % 2 == 0) ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
        }
        if (!nSpecified) {
            n = 64 + fastRand() % 448;
        }
        if (!incxSpecified) {
            incx = 1;
        }
    }

    int64_t lda = n;

    std::cout << "=== STRSV Test Configuration ===" << std::endl;
    std::cout << "  Matrix size (n): " << n << std::endl;
    std::cout << "  LDA: " << lda << std::endl;
    std::cout << "  Random seed: " << g_seed << std::endl;
    std::cout << "  Random range: [-" << randomRange << ", " << randomRange << "]" << std::endl;
    std::cout << "  Diagonal type: " << (diag == ACLBLAS_NON_UNIT ? "NON_UNIT" : "UNIT") << std::endl;
    std::cout << "  Fill mode: " << (uplo == ACLBLAS_LOWER ? "LOWER" : "UPPER") << std::endl;
    std::cout << "  Transpose: " << (trans == ACLBLAS_OP_N ? "N" : "T") << std::endl;
    std::cout << "  incx: " << incx << std::endl;
    std::cout << "================================" << std::endl;

    std::vector<float> A(n * lda, 0.0f);
    std::vector<float> x(n * std::abs(incx), 0.0f);

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            if (uplo == ACLBLAS_LOWER) {
                if (j == i) {
                    A[i * lda + j] = (randomRange * 0.5f) + frand() * randomRange * 0.5f + 0.5f;
                } else {
                    A[i * lda + j] = (frand() - 0.5f) * 2.0f * randomRange;
                }
            } else {
                if (j == i) {
                    A[j * lda + i] = (randomRange * 0.5f) + frand() * randomRange * 0.5f + 0.5f;
                } else {
                    A[j * lda + i] = (frand() - 0.5f) * 2.0f * randomRange;
                }
            }
        }
        x[i * incx] = (frand() - 0.5f) * 2.0f * randomRange;
    }

    std::vector<float> x_original = x;
    std::vector<float> golden = x;

    if (uplo == ACLBLAS_LOWER) {
        ComputeGoldenTrsvLower(diag, trans, n, A, lda, golden);
    } else {
        ComputeGoldenTrsvUpper(diag, trans, n, A, lda, golden);
    }

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    auto ret = aclblasStrsv(nullptr, uplo, trans, diag,
                              n, A.data(), lda, x.data(), incx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasStrsv failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (previewCount > 0) {
        int64_t count = std::min(previewCount, n);
        std::cout << "\n=== Result Preview (first " << count << " elements) ===" << std::endl;
        std::cout << std::setw(12) << "Index" << std::setw(18) << "Golden" << std::setw(18) << "Output" << std::setw(14) << "Diff" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        for (int64_t i = 0; i < count; i++) {
            float diff = std::fabs(x[i * incx] - golden[i]);
            std::cout << std::setw(12) << i
                      << std::setw(18) << std::scientific << std::setprecision(6) << golden[i]
                      << std::setw(18) << x[i * incx]
                      << std::setw(14) << std::fixed << std::setprecision(8) << diff << std::endl;
        }
        std::cout << std::endl;
    }

    return VerifyResult(x, golden, n);
}