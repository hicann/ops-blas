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
 * \file gbmv_perf_test.cpp
 * \brief Performance profiling test for GBMV (General Banded Matrix-Vector multiplication).
 *
 * Measures end-to-end performance for three large-scale scenarios:
 *   - PM-1: m=n=512,  kl=ku=8
 *   - PM-2: m=n=1024, kl=ku=16
 *   - PM-3: m=n=2048, kl=ku=32
 *
 * Uses warm-up iterations and multiple measurement runs for stable timing.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "gbmv_test_utils.h"

struct PerfScenario {
    const char* name;
    int64_t m;
    int64_t n;
    int64_t kl;
    int64_t ku;
    int64_t lda;
};

static const PerfScenario kPerfScenarios[] = {
    {"PM-1 (m=n=512,  kl=ku=8)", 512, 512, 8, 8, 17},
    {"PM-2 (m=n=1024, kl=ku=16)", 1024, 1024, 16, 16, 33},
    {"PM-3 (m=n=2048, kl=ku=32)", 2048, 2048, 32, 32, 65},
};

static constexpr int kNumPerfScenarios = sizeof(kPerfScenarios) / sizeof(kPerfScenarios[0]);
static constexpr int kWarmupRuns = 10;
static constexpr int kMeasureRuns = 50;

struct PerfDeviceBuffers {
    float* aDev = nullptr;
    float* xDev = nullptr;
    float* yDev = nullptr;

    void FreeAll()
    {
        aclrtFree(aDev);
        aclrtFree(xDev);
        aclrtFree(yDev);
    }
};

static int InitPerfTest(
    int32_t deviceId, const PerfScenario& sc, PerfDeviceBuffers& buf, aclrtStream& stream, aclblasHandle_t& handle)
{
    aclError aclRet = aclInit(nullptr);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", aclRet); return -1);

    aclRet = aclrtSetDevice(deviceId);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", aclRet); aclFinalize(); return -1);

    size_t aSize = static_cast<size_t>(sc.lda) * static_cast<size_t>(sc.n);
    size_t xSize = static_cast<size_t>(sc.n);
    size_t ySize = static_cast<size_t>(sc.m);

    aclRet = aclrtMalloc(reinterpret_cast<void**>(&buf.aDev), aSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDev failed\n"); return -1);
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&buf.xDev), xSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDev failed\n"); return -1);
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&buf.yDev), ySize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDev failed\n"); return -1);

    aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed\n"); return -1);

    aclblasStatus_t blasRet = aclblasCreate(&handle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed\n"); return -1);
    blasRet = aclblasSetStream(handle, stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed\n"); return -1);

    return 0;
}

static int CopyHostToDevice(
    const PerfScenario& sc, const PerfDeviceBuffers& buf, const std::vector<float>& aHost,
    const std::vector<float>& xHost, const std::vector<float>& yHost)
{
    size_t aSize = static_cast<size_t>(sc.lda) * static_cast<size_t>(sc.n);
    size_t xSize = static_cast<size_t>(sc.n);
    size_t ySize = static_cast<size_t>(sc.m);

    aclError aclRet =
        aclrtMemcpy(buf.aDev, aSize * sizeof(float), aHost.data(), aSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return -1);
    aclRet =
        aclrtMemcpy(buf.xDev, xSize * sizeof(float), xHost.data(), xSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return -1);
    aclRet =
        aclrtMemcpy(buf.yDev, ySize * sizeof(float), yHost.data(), ySize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return -1);
    return 0;
}

static void PrintPerfStats(const PerfScenario& sc, const std::vector<double>& timesUs)
{
    double sumUs = 0.0;
    for (auto t : timesUs)
        sumUs += t;
    double avgUs = sumUs / timesUs.size();

    double sumSq = 0.0;
    for (auto t : timesUs)
        sumSq += (t - avgUs) * (t - avgUs);
    double stdUs = std::sqrt(sumSq / timesUs.size());

    size_t aBytes = static_cast<size_t>(sc.lda) * static_cast<size_t>(sc.n) * sizeof(float);
    size_t xBytes = static_cast<size_t>(sc.n) * sizeof(float);
    size_t yBytes = static_cast<size_t>(sc.m) * sizeof(float);
    size_t totalBytes = aBytes + xBytes + 2 * yBytes;

    size_t nzCount = 0;
    for (int64_t j = 0; j < sc.n; j++) {
        int64_t iStart = (j > sc.ku) ? (j - sc.ku) : 0;
        int64_t iEnd = (j + sc.kl < sc.m - 1) ? (j + sc.kl) : (sc.m - 1);
        nzCount += static_cast<size_t>(iEnd - iStart + 1);
    }
    size_t nzBytes = nzCount * sizeof(float);
    size_t flops = 2 * nzCount + static_cast<size_t>(sc.m);

    double bwGBpsMem = totalBytes / avgUs;
    double bwGBpsNz = nzBytes / avgUs;
    double gflops = flops / (avgUs * 1e3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Results:" << std::endl;
    std::cout << "    Total data moved (incl zeros): " << totalBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "    Nonzero data moved (band only): " << nzBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "    FLOPs: " << flops / 1e6 << " MFLOPs" << std::endl;
    std::cout << "  Timing (us):" << std::endl;
    std::cout << "    Min: " << timesUs.front() << " us" << std::endl;
    std::cout << "    Avg: " << avgUs << " us" << std::endl;
    std::cout << "    Median: " << timesUs[timesUs.size() / 2] << " us" << std::endl;
    std::cout << "    Max: " << timesUs.back() << " us" << std::endl;
    std::cout << "    Std: " << stdUs << " us" << std::endl;
    std::cout << "  Bandwidth (mem footprint): " << bwGBpsMem << " GB/s" << std::endl;
    std::cout << "  Bandwidth (nonzero data):   " << bwGBpsNz << " GB/s" << std::endl;
    std::cout << "  Compute: " << gflops << " GFLOP/s" << std::endl;
}

static int run_perf_scenario(const PerfScenario& sc)
{
    std::cout << "\n============================================" << std::endl;
    std::cout << "  " << sc.name << std::endl;
    std::cout << "============================================" << std::endl;

    int32_t deviceId = 0;
    std::mt19937 rng(20260516);

    size_t aSize = static_cast<size_t>(sc.lda) * static_cast<size_t>(sc.n);
    std::vector<float> aHost(aSize);
    std::vector<float> xHost(static_cast<size_t>(sc.n));
    std::vector<float> yHost(static_cast<size_t>(sc.m));

    fill_banded_matrix(aHost.data(), sc.m, sc.n, sc.kl, sc.ku, sc.lda, rng);
    fill_contiguous_vector(xHost.data(), sc.n, rng);
    fill_contiguous_vector(yHost.data(), sc.m, rng);

    PerfDeviceBuffers buf;
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    if (InitPerfTest(deviceId, sc, buf, stream, handle) != 0)
        return -1;
    if (CopyHostToDevice(sc, buf, aHost, xHost, yHost) != 0)
        return -1;

    float alpha = 1.0f, beta = 0.5f;

    // Warm-up
    std::cout << "  Warming up (" << kWarmupRuns << " iterations)..." << std::endl;
    size_t ySize = static_cast<size_t>(sc.m);
    for (int i = 0; i < kWarmupRuns; i++) {
        aclrtMemcpy(buf.yDev, ySize * sizeof(float), yHost.data(), ySize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        aclblasSgbmv(
            handle, ACLBLAS_OP_N, sc.m, sc.n, sc.kl, sc.ku, &alpha, buf.aDev, sc.lda, buf.xDev, 1, &beta, buf.yDev, 1);
        aclrtSynchronizeStream(stream);
    }

    // Measurement
    std::cout << "  Measuring (" << kMeasureRuns << " iterations)..." << std::endl;
    std::vector<double> timesUs;
    timesUs.reserve(kMeasureRuns);

    for (int i = 0; i < kMeasureRuns; i++) {
        aclrtMemcpy(buf.yDev, ySize * sizeof(float), yHost.data(), ySize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        auto tStart = std::chrono::high_resolution_clock::now();
        aclblasSgbmv(
            handle, ACLBLAS_OP_N, sc.m, sc.n, sc.kl, sc.ku, &alpha, buf.aDev, sc.lda, buf.xDev, 1, &beta, buf.yDev, 1);
        aclrtSynchronizeStream(stream);
        auto tEnd = std::chrono::high_resolution_clock::now();

        timesUs.push_back(std::chrono::duration<double, std::micro>(tEnd - tStart).count());
    }

    std::sort(timesUs.begin(), timesUs.end());
    PrintPerfStats(sc, timesUs);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    buf.FreeAll();
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}

int32_t main(int32_t argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::cout << "============================================" << std::endl;
    std::cout << "  GBMV Performance Profiling Test" << std::endl;
    std::cout << "  Warmup: " << kWarmupRuns << ", Measure: " << kMeasureRuns << std::endl;
    std::cout << "============================================" << std::endl;

    for (int i = 0; i < kNumPerfScenarios; i++) {
        int ret = run_perf_scenario(kPerfScenarios[i]);
        if (ret != 0) {
            std::cout << "[FATAL] Performance scenario failed: " << kPerfScenarios[i].name << std::endl;
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Performance profiling complete." << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
