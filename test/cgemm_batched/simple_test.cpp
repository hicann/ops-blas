/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Simple test for cgemm_batched kernel
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <complex>
#include "acl/acl.h"

#define GM_ADDR uint8_t*

extern void cgemm_batched_kernel_do(GM_ADDR sync, GM_ADDR a, GM_ADDR b, GM_ADDR gatherOffset, GM_ADDR c,
                                     GM_ADDR workSpace, GM_ADDR tilingGm,
                                     uint32_t numBlocks, void *stream);

int main()
{
    std::cout << "Starting simple cgemm_batched test..." << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 16;
    constexpr int64_t K = 16;
    constexpr int64_t N = 16;
    constexpr int64_t batchCount = 4;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    std::cout << "ACL initialized successfully" << std::endl;

    size_t aSize = batchCount * M * K * 2 * sizeof(float);
    size_t bSize = batchCount * K * N * 2 * sizeof(float);
    size_t cSize = batchCount * M * N * 2 * sizeof(float);
    size_t gatherSize = 1024 * sizeof(uint32_t);
    size_t syncSize = 64 * 1024;

    int64_t useCubeCores = 4;
    int64_t miniBatch = 1;
    size_t workspaceSize = 2 * 2 * miniBatch * K * N * 2 * sizeof(float) * useCubeCores;

    uint8_t* ADevice = nullptr;
    uint8_t* BDevice = nullptr;
    uint8_t* CDevice = nullptr;
    uint8_t* gatherDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    uint8_t* syncDevice = nullptr;

    std::cout << "Allocating device memory..." << std::endl;
    aclrtMalloc((void**)&syncDevice, syncSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&BDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&CDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, 6 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    std::cout << "Preparing tiling data..." << std::endl;
    int64_t tilingData[6] = {M, K, N, batchCount, (batchCount + useCubeCores - 1) / useCubeCores, miniBatch};
    aclrtMemcpy(tilingDevice, sizeof(tilingData), tilingData, sizeof(tilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    uint32_t gatherData[1024];
    for (int i = 0; i < 1024; i++) {
        gatherData[i] = i * 2;
    }
    aclrtMemcpy(gatherDevice, gatherSize, gatherData, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE);

    std::cout << "Calling kernel..." << std::endl;
    std::cout << "  sync addr: " << (void*)syncDevice << std::endl;
    std::cout << "  numBlocks: " << useCubeCores << std::endl;

    cgemm_batched_kernel_do(syncDevice, ADevice, BDevice, gatherDevice, CDevice, 
                             workspaceDevice, tilingDevice, useCubeCores, stream);

    std::cout << "Kernel launched, synchronizing..." << std::endl;
    aclError ret = aclrtSynchronizeStream(stream);
    std::cout << "Sync result: " << ret << std::endl;

    if (ret == ACL_SUCCESS) {
        std::cout << "Kernel executed successfully!" << std::endl;
    } else {
        std::cout << "Kernel execution failed with error: " << ret << std::endl;
    }

    aclrtFree(syncDevice);
    aclrtFree(ADevice);
    aclrtFree(BDevice);
    aclrtFree(CDevice);
    aclrtFree(gatherDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "Test completed" << std::endl;
    return ret == ACL_SUCCESS ? 0 : 1;
}