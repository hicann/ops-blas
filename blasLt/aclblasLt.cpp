/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <securec.h>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "matmul_get_tiling.h"
#include "matmul_kernel.h"
#include "matmul_mxfp4_host.h"

#define GM_ADDR uint8_t*

namespace {

inline aclblasStatus_t CheckedMemcpyS(void* dest, size_t destMax, const void* src, size_t count)
{
    if (memcpy_s(dest, destMax, src, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CheckedMemsetS(void* dest, size_t destMax, size_t count)
{
    if (memset_s(dest, destMax, 0, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool MemcpySSucceeds(void* dest, size_t destMax, const void* src, size_t count)
{
    return memcpy_s(dest, destMax, src, count) == EOK;
}


constexpr int ACLBLASLT_VERSION_MAJOR = 1;
constexpr int ACLBLASLT_VERSION_MINOR = 0;
constexpr int ACLBLASLT_VERSION_PATCH = 0;

constexpr uint32_t ACLBLASLT_HANDLE_MAGIC = 0xACBA1234;
constexpr uint32_t ACLBLASLT_LAYOUT_MAGIC = 0xACBB1234;
constexpr uint32_t ACLBLASLT_DESC_MAGIC = 0xACBC1234;
constexpr uint32_t ACLBLASLT_ALGO_MAGIC = 0xACBD1234;

constexpr size_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;
constexpr size_t L1_SIZE = 512 * 1024;
constexpr size_t L0_SIZE = 256;
constexpr uint32_t DEFAULT_AI_CORES = 8;
constexpr double DEFAULT_PEAK_TFLOPS = 140.0;
constexpr double DEFAULT_PEAK_GBPS = 900.0;

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct Ascend950 {
    static constexpr uint32_t BIAS_SIZE = 4 * 1024;
    static constexpr uint32_t FIXBUF_SIZE = 16 * 1024;
    static constexpr uint32_t UB_SIZE = 248 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 256 * 1024;
};

enum DispatchPolicyType : uint8_t
{
    DISPATCH_POLICY_MMAD_SYNC = 0,
    DISPATCH_POLICY_MMAD_PINGPONG = 1,
    DISPATCH_POLICY_MMAD_MULTI_STAGE = 2,
};

struct AlgoKey {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    aclDataType aType = ACL_FLOAT;
    aclDataType bType = ACL_DT_UNDEFINED;
    aclDataType cType = ACL_DT_UNDEFINED;
    aclDataType dType = ACL_DT_UNDEFINED;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    bool transA = false;
    bool transB = false;

    bool operator==(const AlgoKey& other) const
    {
        return m == other.m && n == other.n && k == other.k && aType == other.aType && bType == other.bType &&
               cType == other.cType && dType == other.dType && computeType == other.computeType &&
               epilogue == other.epilogue && transA == other.transA && transB == other.transB;
    }
};

struct AlgoKeyHasher {
    size_t operator()(const AlgoKey& x) const
    {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2)); };
        mix(x.m);
        mix(x.n);
        mix(x.k);
        mix(static_cast<uint64_t>(x.aType));
        mix(static_cast<uint64_t>(x.bType));
        mix(static_cast<uint64_t>(x.cType));
        mix(static_cast<uint64_t>(x.dType));
        mix(static_cast<uint64_t>(x.computeType));
        mix(static_cast<uint64_t>(x.epilogue));
        mix(static_cast<uint64_t>(x.transA));
        mix(static_cast<uint64_t>(x.transB));
        return h;
    }
};

struct CacheEntry {
    aclblasLtMatmulAlgo_t algo;
    std::list<AlgoKey>::iterator lruIter;
};

struct aclblasLtHandle {
    uint32_t magic = ACLBLASLT_HANDLE_MAGIC;
    bool initialized = false;
    // version info
    int versionMajor = ACLBLASLT_VERSION_MAJOR;
    int versionMinor = ACLBLASLT_VERSION_MINOR;
    // AscendCL runtime
    aclrtContext context = nullptr;
    aclrtStream defaultStream = nullptr;
    int32_t deviceId = 0;
    // workspace
    void* internalWorkspace = nullptr;
    size_t workspaceSize = 0;
    // thread safety
    std::mutex* mutex = nullptr;
    // soc spec
    int npuArch = 0;
    size_t maxSharedMemory = 0;
    // algo cache
    std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>* algoCache = nullptr;
    size_t algoCacheMaxSize = 128;
    std::list<AlgoKey>* lruList = nullptr;
};

struct aclblasLtMatrixLayoutImpl {
    uint32_t magic;
    aclDataType type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_COL;
    int32_t batchCount = 1;
    int64_t stridedBatchOffset = 0;
};
static_assert(
    sizeof(aclblasLtMatrixLayoutImpl) <= sizeof(aclblasLtMatrixLayoutOpaque_t),
    "Impl of aclblasLtMatrixLayout must fit in capsule!");

struct aclblasLtMatmulDescImpl {
    uint32_t magic;
    aclblasComputeType_t computeType;
    aclDataType scaleType;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    const void* bias = nullptr;
    aclDataType biasDataType = ACL_DT_UNDEFINED;
    const void* scaleA = nullptr;
    const void* scaleB = nullptr;
};

constexpr size_t kBiasPtrStorageBytes = sizeof(void*);
static_assert(
    sizeof(aclblasLtMatmulDescImpl) <= sizeof(aclblasLtMatmulDescOpaque_t),
    "Impl of aclblasLtMatmulDesc must fit in capsule!");

struct aclblasLtMatmulPreferenceImpl {
    uint32_t magic;
    uint32_t searchMode = 0;
    size_t maxWorkspaceBytes = DEFAULT_WORKSPACE_SIZE;
    int32_t maxResults = 3;
    bool allowMixedPrecision = true;
    bool allowSplitK = true;
    // tiling
    uint32_t preferredL0M = 0;
    uint32_t preferredL0N = 0;
    uint32_t preferredL0K = 0;
    // Scheduling
    bool preferPingpong = false;
    bool preferDoubleBuffer = false;
    float minEfficiency = 0.5f;
};
static_assert(
    sizeof(aclblasLtMatmulPreferenceImpl) <= sizeof(aclblasLtMatmulPreferenceOpaque_t),
    "Impl of aclblasLtMatmulPreference must fit in capsule!");

struct AscendHardwareCaps {
    uint32_t numAICores = DEFAULT_AI_CORES;
    uint32_t l0CubeSize = L0_SIZE;
    size_t l1BufferSize = L1_SIZE;
    double memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
    double peakTFlops = DEFAULT_PEAK_TFLOPS;
    double bandwidthBoundThreshold = 32.0;
};

struct AlgoCandidate {
    uint32_t algoId = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 128;
    uint32_t l1TileK = 128;
    uint32_t l0TileM = 64;
    uint32_t l0TileN = 64;
    uint32_t l0TileK = 64;
    DispatchPolicyType policy = DISPATCH_POLICY_MMAD_SYNC;
    uint32_t numBuffers = 1;
    uint32_t splitKFactor = 1;
    size_t workspaceSize = 0;
    double peakPerformance = DEFAULT_PEAK_TFLOPS;
};

struct ScoredResult {
    AlgoCandidate cand;
    double estimatedTimeMs = 0.0;
    double totalScore = 0.0;
    bool isEfficient = true;
};

struct PackedAlgo {
    uint32_t magic;
    uint32_t algoId;
    uint16_t l1mDiv16;
    uint16_t l1nDiv16;
    uint8_t policy;
    uint8_t numBuffers;
    uint8_t splitK;
    uint8_t flags;
};
static_assert(sizeof(PackedAlgo) == 16, "PackedAlgo must fit algo.data");

template <typename T>
static aclblasStatus_t AllocHandle(T** out)
{
    if (out == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *out = nullptr;
    T* p = new (std::nothrow) T();
    if (p == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    *out = p;
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
static aclblasStatus_t FreeHandle(T* p)
{
    if (p == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    delete p;
    return ACLBLAS_STATUS_SUCCESS;
}

static size_t GetTypeSize(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_INT8:
            return 1;
        default:
            return 0;
    }
}

static bool IsDataTypeSupported(aclDataType dt)
{
    return GetTypeSize(dt) != 0;
}

static bool IsMxfp8Type(aclDataType dt)
{
    return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2;
}

static bool IsMxfp4Type(aclDataType dt)
{
    return dt == ACL_FLOAT4_E2M1;
}

static bool CheckComputeTypeCompatibility(aclblasComputeType_t ct, aclDataType typeA, aclDataType typeB)
{
    // FP32 path: A and B must both be FP32
    if (typeA == ACL_FLOAT && typeB == ACL_FLOAT) {
        return ct == ACLBLAS_COMPUTE_32F || ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_TF32;
    }
    // MXFP8 path: A and B must both be MXFP8 series (allow E4M3FN/E5M2 mixed)
    if (IsMxfp8Type(typeA) && IsMxfp8Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    // MXFP4 path: A and B must both be E2M1
    if (IsMxfp4Type(typeA) && IsMxfp4Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    // FP16 legacy path
    if (typeA == ACL_FLOAT16 && typeB == ACL_FLOAT16) {
        return ct == ACLBLAS_COMPUTE_16F || ct == ACLBLAS_COMPUTE_16F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F ||
               ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_16F;
    }
    // INT8 legacy path
    if (typeA == ACL_INT8 && typeB == ACL_INT8) {
        return ct == ACLBLAS_COMPUTE_32I || ct == ACLBLAS_COMPUTE_32I_PEDANTIC;
    }
    return false;
}

static uint32_t CeilDivU64(uint64_t a, uint64_t b)
{
    return static_cast<uint32_t>((a + b - 1) / b);
}

static uint32_t GenerateAlgoId(
    DispatchPolicyType policy, uint32_t l1m, uint32_t l1n, uint32_t l1k, uint32_t splitKFactor)
{
    return (static_cast<uint32_t>(policy) << 28) ^ (l1m << 16) ^ (l1n << 8) ^ (l1k << 2) ^ splitKFactor;
}

static aclblasLtMatmulAlgo_t BuildAlgoFromCandidate(const AlgoCandidate& cand)
{
    aclblasLtMatmulAlgo_t out{};
    PackedAlgo packed{};
    packed.magic = ACLBLASLT_ALGO_MAGIC;
    packed.algoId = cand.algoId;
    packed.l1mDiv16 = static_cast<uint16_t>(cand.l1TileM / 16);
    packed.l1nDiv16 = static_cast<uint16_t>(cand.l1TileN / 16);
    packed.policy = static_cast<uint8_t>(cand.policy);
    packed.numBuffers = static_cast<uint8_t>(cand.numBuffers);
    packed.splitK = static_cast<uint8_t>(cand.splitKFactor);
    (void)MemcpySSucceeds(out.data, sizeof(out.data), &packed, sizeof(packed));
    out.max_workspace_bytes = cand.workspaceSize;
    return out;
}

static bool DecodeAlgo(const aclblasLtMatmulAlgo_t& algo, PackedAlgo* packed)
{
    if (packed == nullptr) {
        return false;
    }
    if (!MemcpySSucceeds(packed, sizeof(PackedAlgo), algo.data, sizeof(PackedAlgo))) {
        return false;
    }
    return packed->magic == ACLBLASLT_ALGO_MAGIC;
}

static void GetAscendHardwareCaps(int32_t, AscendHardwareCaps* caps)
{
    if (caps == nullptr) {
        return;
    }
    // 当前仓库保持轻量默认能力值，后续可对接真实设备查询。
    caps->numAICores = DEFAULT_AI_CORES;
    caps->l0CubeSize = L0_SIZE;
    caps->l1BufferSize = L1_SIZE;
    caps->memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
    caps->peakTFlops = DEFAULT_PEAK_TFLOPS;
    caps->bandwidthBoundThreshold = 32.0;
}

static void SelectL1TileShape(
    uint64_t m, uint64_t n, uint64_t, uint32_t numAICores, uint32_t prefL0M, uint32_t prefL0N, uint32_t prefL0K,
    uint32_t* l1M, uint32_t* l1N, uint32_t* l1K)
{
    const uint32_t candidates[][3] = {
        {128, 256, 256}, {256, 128, 256}, {256, 256, 128}, {128, 128, 128}, {256, 256, 64}};

    float bestScore = -1.0f;
    const uint32_t* best = candidates[0];

    for (const auto& cand : candidates) {
        uint32_t cm = cand[0];
        uint32_t cn = cand[1];
        uint32_t ck = cand[2];

        uint32_t tilesM = CeilDivU64(m, cm);
        uint32_t tilesN = CeilDivU64(n, cn);
        uint32_t totalTiles = tilesM * tilesN;

        float balanceScore =
            1.0f - static_cast<float>(totalTiles % std::max(1u, numAICores)) / std::max(1u, numAICores);
        size_t l1Usage = static_cast<size_t>(cm) * ck * sizeof(float) + static_cast<size_t>(cn) * ck * sizeof(float);
        float l1Util = static_cast<float>(l1Usage) / static_cast<float>(L1_SIZE);

        float l0Match = 0.0f;
        if (prefL0M > 0 && cm % prefL0M == 0) {
            l0Match += 0.3f;
        }
        if (prefL0N > 0 && cn % prefL0N == 0) {
            l0Match += 0.3f;
        }
        if (prefL0K > 0 && ck % prefL0K == 0) {
            l0Match += 0.4f;
        }

        float score = balanceScore * 0.4f + std::min(l1Util, 1.0f) * 0.3f + l0Match * 0.3f;
        if (score > bestScore) {
            bestScore = score;
            best = cand;
        }
    }

    *l1M = best[0];
    *l1N = best[1];
    *l1K = best[2];
}

static void SelectL0TileShape(
    uint32_t l1M, uint32_t l1N, uint32_t l1K, size_t, size_t, aclDataType, aclDataType, uint32_t* l0M, uint32_t* l0N,
    uint32_t* l0K)
{
    *l0K = std::min(64u, l1K);
    *l0M = std::min(128u, l1M);
    *l0N = std::min(256u, l1N);

    while (*l0M > 16 && (l1M % *l0M != 0)) {
        --(*l0M);
    }
    while (*l0N > 16 && (l1N % *l0N != 0)) {
        --(*l0N);
    }
}

static uint32_t SelectSplitKForAscend(uint32_t l1LoopsK, uint32_t numAICores)
{
    if (numAICores == 0) {
        return 1;
    }
    uint32_t candidate = std::min(l1LoopsK, numAICores);
    return std::max(1u, candidate);
}

static size_t CalculateWorkspaceForAscend(uint64_t m, uint64_t n, uint32_t splitKFactor, aclblasLtEpilogue_t epilogue)
{
    size_t workspace = 0;
    if (splitKFactor > 1) {
        workspace +=
            static_cast<size_t>(splitKFactor) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    }

    switch (epilogue) {
        case ACLBLASLT_EPILOGUE_BIAS:
        case ACLBLASLT_EPILOGUE_RELU_BIAS:
        case ACLBLASLT_EPILOGUE_GELU_BIAS:
            workspace += static_cast<size_t>(m) * sizeof(float);
            break;
        case ACLBLASLT_EPILOGUE_GELU:
        case ACLBLASLT_EPILOGUE_RELU:
            workspace += 64 * 1024;
            break;
        default:
            break;
    }
    return workspace;
}

static bool CheckHandleValid(const aclblasLtHandle* h)
{
    return h != nullptr && h->magic == ACLBLASLT_HANDLE_MAGIC && h->initialized && h->algoCache != nullptr &&
           h->lruList != nullptr;
}

static bool BuildGemmShape(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* A, const aclblasLtMatrixLayoutImpl* B,
    const aclblasLtMatrixLayoutImpl* D, uint64_t* m, uint64_t* n, uint64_t* k)
{
    if (desc == nullptr || A == nullptr || B == nullptr || D == nullptr || m == nullptr || n == nullptr ||
        k == nullptr) {
        return false;
    }
    const bool transA = (desc->transA != ACLBLAS_OP_N);
    const bool transB = (desc->transB != ACLBLAS_OP_N);

    const uint64_t mA = transA ? A->cols : A->rows;
    const uint64_t kA = transA ? A->rows : A->cols;
    const uint64_t kB = transB ? B->cols : B->rows;
    const uint64_t nB = transB ? B->rows : B->cols;

    if (mA != D->rows || kA != kB || nB != D->cols) {
        return false;
    }

    *m = mA;
    *n = nB;
    *k = kA;
    return true;
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtGetVersion(size_t* version)
{
    if (version == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *version = (static_cast<size_t>(ACLBLASLT_VERSION_MAJOR) << 24) |
               (static_cast<size_t>(ACLBLASLT_VERSION_MINOR) << 16) | static_cast<size_t>(ACLBLASLT_VERSION_PATCH);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value)
{
    if (value == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    switch (type) {
        case ACLBLASLT_PROPERTY_MAJOR_VERSION:
            *value = ACLBLASLT_VERSION_MAJOR;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_PROPERTY_MINOR_VERSION:
            *value = ACLBLASLT_VERSION_MINOR;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_PROPERTY_PATCH_LEVEL:
            *value = ACLBLASLT_VERSION_PATCH;
            return ACLBLAS_STATUS_SUCCESS;
        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
}

aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtHandle* h = nullptr;
    auto st = AllocHandle(&h);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    int32_t deviceId = 0;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    aclrtContext currentCtx = nullptr;
    aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    h->deviceId = deviceId;
    h->context = currentCtx;
    h->defaultStream = nullptr;
    h->workspaceSize = DEFAULT_WORKSPACE_SIZE;
    h->internalWorkspace = std::malloc(h->workspaceSize);
    if (h->internalWorkspace == nullptr) {
        delete h;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->mutex = new (std::nothrow) std::mutex();
    h->algoCache = new (std::nothrow) std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>();
    h->lruList = new (std::nothrow) std::list<AlgoKey>();
    if (h->mutex == nullptr || h->algoCache == nullptr || h->lruList == nullptr) {
        delete h->mutex;
        delete h->algoCache;
        delete h->lruList;
        std::free(h->internalWorkspace);
        delete h;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->npuArch = 2;
    h->maxSharedMemory = L1_SIZE;
    h->initialized = true;
    *lightHandle = reinterpret_cast<aclblasLtHandle_t>(h);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = reinterpret_cast<aclblasLtHandle*>(lightHandle);
    if (h->magic != ACLBLASLT_HANDLE_MAGIC) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    h->initialized = false;
    delete h->mutex;
    delete h->algoCache;
    delete h->lruList;
    std::free(h->internalWorkspace);
    h->mutex = nullptr;
    h->algoCache = nullptr;
    h->lruList = nullptr;
    h->internalWorkspace = nullptr;
    h->workspaceSize = 0;
    return FreeHandle(h);
}

aclblasStatus_t aclblasLtMatrixLayoutCreate(
    aclblasLtMatrixLayout_t* layout, aclDataType type, uint64_t rows, uint64_t cols, int64_t ld)
{
    // 1. 参数校验（BLAS/cuBLAS 允许 m=0 或 n=0 的空矩阵，仅拒绝非法 ld）
    if (layout == nullptr || ld < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *layout = nullptr;
    // 2. 堆上分配胶囊
    auto* capsule = new (std::nothrow) aclblasLtMatrixLayoutOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // 3. 栈上创建Impl，初始化后拷贝进胶囊
    aclblasLtMatrixLayoutImpl impl;
    impl.magic = ACLBLASLT_LAYOUT_MAGIC;
    impl.type = type;
    impl.rows = rows;
    impl.cols = cols;
    impl.ld = (ld == 0) ? static_cast<int64_t>(rows) : ld;
    // 4. Impl → 胶囊
    static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatrixLayoutImpl too large, not fit in capsule!");
    aclblasStatus_t copyStatus = CheckedMemcpyS(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }
    if (sizeof(*capsule) > sizeof(impl)) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(capsule) + sizeof(impl), sizeof(*capsule) - sizeof(impl),
            sizeof(*capsule) - sizeof(impl));
        if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
            delete capsule;
            return copyStatus;
        }
    }
    // 5. 返回胶囊指针
    *layout = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t layout)
{
    if (layout == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatrixLayoutOpaque_t*>(layout);
    delete capsule;

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(
    aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (layout == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // 解包到栈上
    aclblasLtMatrixLayoutImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), layout, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATRIX_LAYOUT_TYPE:
            if (sizeInBytes != sizeof(impl.type)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.type = *reinterpret_cast<const aclDataType*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_ROWS:
            if (sizeInBytes != sizeof(impl.rows)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.rows = *reinterpret_cast<const uint64_t*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_COLS:
            if (sizeInBytes != sizeof(impl.cols)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.cols = *reinterpret_cast<const uint64_t*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_LD:
            if (sizeInBytes != sizeof(impl.ld)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.ld = *reinterpret_cast<const int64_t*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_ORDER:
            if (sizeInBytes != sizeof(impl.order)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.order = *reinterpret_cast<const aclblasLtOrder_t*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            if (sizeInBytes != sizeof(impl.batchCount)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.batchCount = *reinterpret_cast<const int32_t*>(buf);
            break;

        case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            if (sizeInBytes != sizeof(impl.stridedBatchOffset)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.stridedBatchOffset = *reinterpret_cast<const int64_t*>(buf);
            break;

        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // 压缩回堆上
    copyStatus = CheckedMemcpyS(layout, sizeof(*layout), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (sizeof(*layout) > sizeof(impl)) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(layout) + sizeof(impl), sizeof(*layout) - sizeof(impl),
            sizeof(*layout) - sizeof(impl));
        if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
            return copyStatus;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(
    const aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (layout == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatrixLayoutImpl impl;
    static_assert(sizeof(impl) <= sizeof(*layout), "aclblasLtMatrixLayoutImpl too large for capsule");
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), layout, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t actualSize = 0;

    switch (attr) {
        case ACLBLASLT_MATRIX_LAYOUT_TYPE:
            actualSize = sizeof(impl.type);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<aclDataType*>(buf) = impl.type;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_ROWS:
            actualSize = sizeof(impl.rows);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<uint64_t*>(buf) = impl.rows;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_COLS:
            actualSize = sizeof(impl.cols);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<uint64_t*>(buf) = impl.cols;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_LD:
            actualSize = sizeof(impl.ld);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<int64_t*>(buf) = impl.ld;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_ORDER:
            actualSize = sizeof(impl.order);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<aclblasLtOrder_t*>(buf) = impl.order;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            actualSize = sizeof(impl.batchCount);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<int32_t*>(buf) = impl.batchCount;
            break;

        case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            actualSize = sizeof(impl.stridedBatchOffset);
            if (sizeInBytes < actualSize) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            *reinterpret_cast<int64_t*>(buf) = impl.stridedBatchOffset;
            break;

        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (sizeWritten != nullptr) {
        *sizeWritten = actualSize;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescCreate(
    aclblasLtMatmulDesc_t* desc, aclblasComputeType_t computeType, aclDataType scaleType)
{
    if (desc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *desc = nullptr;

    auto* capsule = new (std::nothrow) aclblasLtMatmulDescOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasLtMatmulDescImpl impl;
    impl.magic = ACLBLASLT_DESC_MAGIC;
    impl.computeType = computeType;
    impl.scaleType = scaleType;

    static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatmulDescImpl too large, not fit in capsule!");
    aclblasStatus_t copyStatus = CheckedMemcpyS(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }
    if (sizeof(*capsule) > sizeof(impl)) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(capsule) + sizeof(impl), sizeof(*capsule) - sizeof(impl),
            sizeof(*capsule) - sizeof(impl));
        if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
            delete capsule;
            return copyStatus;
        }
    }

    *desc = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t desc)
{
    if (desc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatmulDescOpaque_t*>(desc);
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescSetAttribute(
    aclblasLtMatmulDesc_t desc, aclblasLtMatmulDescAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (desc == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), desc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATMUL_DESC_EPILOGUE: {
            if (sizeInBytes != sizeof(aclblasLtEpilogue_t) && sizeInBytes != sizeof(uint32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            uint32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(uint32_t));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.epilogue = static_cast<aclblasLtEpilogue_t>(v);
            break;
        }
        case ACLBLASLT_MATMUL_DESC_BIAS_POINTER: {
            if (sizeInBytes != kBiasPtrStorageBytes) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            void* biasPtr = nullptr;
            copyStatus = CheckedMemcpyS(&biasPtr, kBiasPtrStorageBytes, buf, kBiasPtrStorageBytes);
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.bias = biasPtr;
            break;
        }
        case ACLBLASLT_MATMUL_DESC_TRANSA: {
            if (sizeInBytes != sizeof(int32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            int32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.transA = static_cast<aclblasOperation_t>(v);
            break;
        }
        case ACLBLASLT_MATMUL_DESC_TRANSB: {
            if (sizeInBytes != sizeof(int32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            int32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.transB = static_cast<aclblasOperation_t>(v);
            break;
        }
        case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: {
            if (sizeInBytes != sizeof(int32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            int32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.biasDataType = static_cast<aclDataType>(v);
            break;
        }
        case ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER: {
            if (sizeInBytes != sizeof(void*)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            void* scalePtr = nullptr;
            copyStatus = CheckedMemcpyS(&scalePtr, sizeof(void*), buf, sizeof(void*));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.scaleA = scalePtr;
            break;
        }
        case ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER: {
            if (sizeInBytes != sizeof(void*)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            void* scalePtr = nullptr;
            copyStatus = CheckedMemcpyS(&scalePtr, sizeof(void*), buf, sizeof(void*));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            impl.scaleB = scalePtr;
            break;
        }
        case ACLBLASLT_MATMUL_DESC_A_SCALE_MODE:
        case ACLBLASLT_MATMUL_DESC_B_SCALE_MODE:
            break;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    copyStatus = CheckedMemcpyS(desc, sizeof(*desc), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (sizeof(*desc) > sizeof(impl)) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(desc) + sizeof(impl), sizeof(*desc) - sizeof(impl),
            sizeof(*desc) - sizeof(impl));
        if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
            return copyStatus;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescGetAttribute(
    aclblasLtMatmulDesc_t desc, aclblasLtMatmulDescAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)
{
    if (desc == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), desc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t requiredSize = 0;
    const void* srcPtr = nullptr;

    switch (attr) {
        case ACLBLASLT_MATMUL_DESC_EPILOGUE:
            requiredSize = sizeof(impl.epilogue);
            srcPtr = &impl.epilogue;
            break;

        case ACLBLASLT_MATMUL_DESC_BIAS_POINTER:
            requiredSize = kBiasPtrStorageBytes;
            srcPtr = &impl.bias;
            break;

        case ACLBLASLT_MATMUL_DESC_TRANSA:
            requiredSize = sizeof(impl.transA);
            srcPtr = &impl.transA;
            break;

        case ACLBLASLT_MATMUL_DESC_TRANSB:
            requiredSize = sizeof(impl.transB);
            srcPtr = &impl.transB;
            break;

        case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            requiredSize = sizeof(impl.biasDataType);
            srcPtr = &impl.biasDataType;
            break;

        case ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            requiredSize = sizeof(impl.scaleA);
            srcPtr = &impl.scaleA;
            break;

        case ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            requiredSize = sizeof(impl.scaleB);
            srcPtr = &impl.scaleB;
            break;

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    // 检查用户缓冲区大小
    if (sizeInBytes < requiredSize) {
        if (sizeWritten != nullptr) {
            *sizeWritten = requiredSize;
        }
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    copyStatus = CheckedMemcpyS(buf, sizeInBytes, srcPtr, requiredSize);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    if (sizeWritten != nullptr) {
        *sizeWritten = requiredSize;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *pref = nullptr;
    auto* capsule = new (std::nothrow) aclblasLtMatmulPreferenceOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t copyStatus = CheckedMemsetS(capsule, sizeof(*capsule), sizeof(*capsule));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    aclblasLtMatmulPreferenceImpl impl;
    copyStatus = CheckedMemcpyS(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }
    if (sizeof(*capsule) > sizeof(impl)) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(capsule) + sizeof(impl), sizeof(*capsule) - sizeof(impl),
            sizeof(*capsule) - sizeof(impl));
        if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
            delete capsule;
            return copyStatus;
        }
    }

    *pref = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(pref);
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (pref == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE: {
            if (sizeInBytes != sizeof(uint32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            uint32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(v));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            // 0=heuristic, 1=exhaustive, 2=fast
            if (v > 2) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.searchMode = v;
            break;
        }

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: {
            if (sizeInBytes != sizeof(size_t) && sizeInBytes != sizeof(uint64_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            size_t v = 0;
            const size_t copyBytes = std::min(sizeInBytes, sizeof(v));
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, copyBytes);
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            if (v > INT64_MAX) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.maxWorkspaceBytes = v;
            break;
        }

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    copyStatus = CheckedMemcpyS(pref, sizeof(*pref), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (pref == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t requiredSize = 0;
    const void* srcPtr = nullptr;

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE:
            requiredSize = sizeof(impl.searchMode);
            srcPtr = &impl.searchMode;
            break;

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            requiredSize = sizeof(impl.maxWorkspaceBytes);
            srcPtr = &impl.maxWorkspaceBytes;
            break;

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    if (sizeInBytes < requiredSize) {
        if (sizeWritten != nullptr) {
            *sizeWritten = requiredSize;
        }
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    copyStatus = CheckedMemcpyS(buf, sizeInBytes, srcPtr, requiredSize);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    if (sizeWritten != nullptr) {
        *sizeWritten = requiredSize;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t Adesc,
    aclblasLtMatrixLayout_t Bdesc, aclblasLtMatrixLayout_t Cdesc, aclblasLtMatrixLayout_t Ddesc,
    aclblasLtMatmulPreference_t preference, int requestedAlgoCount,
    aclblasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount)
{
    // Validate input parameters
    if (returnAlgoCount == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *returnAlgoCount = 0;

    if (requestedAlgoCount <= 0 || heuristicResultsArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (lightHandle == nullptr || computeDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Get workspace size from preference
    size_t maxWorkspace = 0;
    if (preference != nullptr) {
        auto* p = reinterpret_cast<aclblasLtMatmulPreferenceImpl*>(preference);
        maxWorkspace = p->maxWorkspaceBytes;
    }

    // Get matrix dimensions
    auto* A = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* B = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* D = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);

    // Validate dimensions for GEMM: D = A * B + C
    // A: m x k, B: k x n, C/D: m x n
    uint64_t m = D->rows;
    uint64_t n = D->cols;
    uint64_t k = (desc->transA == ACLBLAS_OP_N) ? A->cols : A->rows;

    // Basic validation
    if (!CheckComputeTypeCompatibility(desc->computeType, A->type, B->type)) {
        heuristicResultsArray[0].state = ACLBLAS_STATUS_INVALID_VALUE;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Fill heuristic result
    heuristicResultsArray[0].algo.max_workspace_bytes = maxWorkspace;
    heuristicResultsArray[0].workspaceSize = maxWorkspace;
    heuristicResultsArray[0].state = ACLBLAS_STATUS_SUCCESS;
    heuristicResultsArray[0].wavesCount = 1.0f;
    if (CheckedMemsetS(
            heuristicResultsArray[0].reserved, sizeof(heuristicResultsArray[0].reserved),
            sizeof(heuristicResultsArray[0].reserved)) != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    *returnAlgoCount = 1;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmul(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, const void* alpha, const void* A,
    aclblasLtMatrixLayout_t Adesc, const void* B, aclblasLtMatrixLayout_t Bdesc, const void* beta, const void* C,
    aclblasLtMatrixLayout_t Cdesc, void* D, aclblasLtMatrixLayout_t Ddesc, const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, aclrtStream stream)
{
    // Validate lightHandle
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    // Validate descriptors
    if (computeDesc == nullptr || Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate pointers
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Get layout info
    auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* CLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc);
    auto* DLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);

    // Get dimensions
    uint64_t m = DLayout->rows;
    uint64_t n = DLayout->cols;
    uint64_t k = 0;

    // Determine k based on transpose operations
    if (desc->transA == ACLBLAS_OP_N) {
        k = ALayout->cols;
    } else {
        k = ALayout->rows;
    }

    // BLAS/cuBLAS convention: m=0 or n=0 is a no-op, succeed without touching matrices.
    if (m == 0U || n == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (A == nullptr || B == nullptr || D == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate workspace alignment (must be 16B aligned)
    if (workspace != nullptr && (reinterpret_cast<uintptr_t>(workspace) & 0xF) != 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate workspace size
    if (algo != nullptr && workspaceSizeInBytes < algo->max_workspace_bytes) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Actual GEMM implementation using ltmatmul routing
    auto* handleImpl = reinterpret_cast<aclblasLtHandle*>(lightHandle);
    int32_t deviceId = handleImpl->deviceId;
    int64_t cubeCoreNum = 0;
    aclError aclRet = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCoreNum);
    if (aclRet != ACL_SUCCESS || cubeCoreNum <= 0) {
        cubeCoreNum = 8;  // Fallback minimum
    }
    uint32_t numBlocks = static_cast<uint32_t>(cubeCoreNum);

    float alphaValue = *(reinterpret_cast<const float*>(alpha));
    float betaValue = *(reinterpret_cast<const float*>(beta));
    bool needEpilogue = (alphaValue != 1.0f || betaValue != 0.0f);
    bool cOverlap = (C == D) && needEpilogue;
    void* dRawAddr = needEpilogue && cOverlap ? workspace : D;

    aclDataType dtypeA = ALayout->type;
    aclDataType dtypeB = BLayout->type;
    aclDataType dtypeD = DLayout->type;
    bool transA = (desc->transA != ACLBLAS_OP_N);
    bool transB = (desc->transB != ACLBLAS_OP_N);

    if ((IsMxfp8Type(dtypeA) || IsMxfp4Type(dtypeA)) && (k % 32 != 0)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // ===== Step 1: MMAD Kernel =====
    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) {
        MatmulFp32TilingData fp32Tiling;
        matmul_fp32_get_tiling(
            m, n, k, transA, transB, static_cast<uint32_t>(ALayout->ld), static_cast<uint32_t>(BLayout->ld), numBlocks,
            fp32Tiling);
        matmul_fp32_kernel_do(
            static_cast<GM_ADDR>(const_cast<void*>(A)),
            static_cast<GM_ADDR>(const_cast<void*>(B)),
            static_cast<GM_ADDR>(dRawAddr),
            fp32Tiling, numBlocks, stream);
    } else if (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) {
        void* scaleA = const_cast<void*>(desc->scaleA);
        void* scaleB = const_cast<void*>(desc->scaleB);
        if (scaleA == nullptr || scaleB == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        QuantMatmulTilingData mxfp8Tiling;
        matmul_mxfp8_get_tiling(m, n, k, transA, transB, numBlocks, mxfp8Tiling);
        matmul_mxfp8_kernel_do(
            static_cast<GM_ADDR>(const_cast<void*>(A)),
            static_cast<GM_ADDR>(const_cast<void*>(B)),
            static_cast<GM_ADDR>(scaleA),
            static_cast<GM_ADDR>(scaleB),
            static_cast<GM_ADDR>(dRawAddr),
            mxfp8Tiling, transA, transB, stream);
    } else if (IsMxfp4Type(dtypeA) && IsMxfp4Type(dtypeB)) {
        void* scaleA = const_cast<void*>(desc->scaleA);
        void* scaleB = const_cast<void*>(desc->scaleB);
        if (scaleA == nullptr || scaleB == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        QuantMatmulTilingData mxfp4Tiling;
        matmul_mxfp4_get_tiling(m, n, k, transA, transB, numBlocks, mxfp4Tiling);
        ltmatmul_mxfp4_kernel_do(
            static_cast<GM_ADDR>(const_cast<void*>(A)),
            static_cast<GM_ADDR>(const_cast<void*>(B)),
            static_cast<GM_ADDR>(scaleA),
            static_cast<GM_ADDR>(scaleB),
            static_cast<GM_ADDR>(dRawAddr),
            mxfp4Tiling, dtypeA, dtypeB, dtypeD, transA, transB, stream);
    } else {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    // ===== Step 2: Epilogue alpha*D_raw + beta*C =====
    if (needEpilogue) {
        uint32_t totalElements = static_cast<uint32_t>(m * n);
        uint32_t dtypeDValue = (dtypeD == ACL_FLOAT) ? 0U : 27U;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"
