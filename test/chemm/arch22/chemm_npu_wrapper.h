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

#include <cstdint>
#include <algorithm>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"

class ChemmTimeoutHandleTracker {
public:
    static bool IsTimedOut(aclblasHandle_t handle)
    {
        std::lock_guard<std::mutex> lock(GetMutex());
        return GetTimedOutHandles().count(handle) != 0U;
    }

    static void MarkTimedOut(aclblasHandle_t handle)
    {
        std::lock_guard<std::mutex> lock(GetMutex());
        (void)GetTimedOutHandles().insert(handle);
    }

private:
    static std::mutex &GetMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_set<aclblasHandle_t> &GetTimedOutHandles()
    {
        static std::unordered_set<aclblasHandle_t> timedOutHandles;
        return timedOutHandles;
    }
};

class ChemmWorkspaceTracker {
public:
    static aclblasStatus_t Ensure(aclblasHandle_t handle, size_t size)
    {
        std::lock_guard<std::mutex> lock(Mutex());
        auto& map = Workspaces();
        auto it = map.find(handle);
        if (it != map.end() && it->second.size >= size) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        void* pointer = nullptr;
        if (aclrtMalloc(&pointer, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclblasStatus_t status = aclblasSetWorkspace(handle, pointer, size);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            (void)aclrtFree(pointer);
            return status;
        }
        map[handle] = {pointer, size};
        return ACLBLAS_STATUS_SUCCESS;
    }

private:
    struct Entry { void* pointer; size_t size; };
    static std::mutex& Mutex() { static std::mutex mutex; return mutex; }
    static std::unordered_map<aclblasHandle_t, Entry>& Workspaces()
    {
        static std::unordered_map<aclblasHandle_t, Entry> workspaces;
        return workspaces;
    }
};

struct ChemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
};

inline void ChemmReleaseDeviceBuffers(ChemmDeviceBuffers& buffers)
{
    (void)aclrtFree(buffers.a);
    (void)aclrtFree(buffers.b);
    (void)aclrtFree(buffers.c);
    buffers = {};
}

inline aclblasStatus_t ChemmAllocAndCopy(void** device, size_t bytes, const void* host)
{
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmRestoreTimeout(uint32_t savedTimeout, bool& changed)
{
    if (!changed) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtSetOpExecuteTimeOutWithMs(savedTimeout) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    changed = false;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmPrepareWorkspace(aclblasHandle_t handle, size_t bytes)
{
    if (bytes <= 4U * 1024U * 1024U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ChemmWorkspaceTracker::Ensure(handle, bytes);
}

inline aclblasStatus_t ChemmRunDeviceOperation(aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, int64_t m, int64_t n, const aclblasComplex* alpha, const aclblasComplex* A,
    int64_t lda, const aclblasComplex* B, int64_t ldb, const aclblasComplex* beta, aclblasComplex* C,
    int64_t ldc, size_t aBytes, size_t bBytes, size_t cBytes, size_t workspaceBytes)
{
    ChemmDeviceBuffers buffers;
    aclblasStatus_t status = ChemmAllocAndCopy(&buffers.a, aBytes, A);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = ChemmAllocAndCopy(&buffers.b, bBytes, B);
    if (status != ACLBLAS_STATUS_SUCCESS) { ChemmReleaseDeviceBuffers(buffers); return status; }
    status = ChemmAllocAndCopy(&buffers.c, cBytes, C);
    if (status != ACLBLAS_STATUS_SUCCESS) { ChemmReleaseDeviceBuffers(buffers); return status; }
    status = ChemmPrepareWorkspace(handle, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) { ChemmReleaseDeviceBuffers(buffers); return status; }

    uint32_t savedTimeout = 0U;
    bool timeoutChanged = false;
    if (aclrtGetOpExecuteTimeout(&savedTimeout) != ACL_SUCCESS ||
        aclrtSetOpExecuteTimeOutWithMs(240000U) != ACL_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    timeoutChanged = true;
    aclblasStatus_t ret = aclblasChemm(handle, side, uplo, m, n, alpha,
        static_cast<const aclblasComplex*>(buffers.a), lda, static_cast<const aclblasComplex*>(buffers.b), ldb,
        beta, static_cast<aclblasComplex*>(buffers.c), ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDeviceWithTimeout(270000U) == ACL_SUCCESS) {
        if (aclrtMemcpy(C, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        ChemmTimeoutHandleTracker::MarkTimedOut(handle);
        ret = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclblasStatus_t restore = ChemmRestoreTimeout(savedTimeout, timeoutChanged);
    ChemmReleaseDeviceBuffers(buffers);
    if (restore != ACLBLAS_STATUS_SUCCESS) return restore;
    return ret;
}

inline aclblasStatus_t aclblasChemm_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    if (handle != nullptr && ChemmTimeoutHandleTracker::IsTimedOut(handle)) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (handle == nullptr || m <= 0 || n <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return aclblasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aElemCount = static_cast<size_t>(aDim) * static_cast<size_t>(lda);
    const size_t bElemCount = static_cast<size_t>(m) * static_cast<size_t>(ldb);
    const size_t cElemCount = static_cast<size_t>(m) * static_cast<size_t>(ldc);
    const size_t aBytes = aElemCount * 2U * sizeof(float);
    const size_t bBytes = bElemCount * 2U * sizeof(float);
    const size_t cBytes = cElemCount * 2U * sizeof(float);

    const uint64_t k = static_cast<uint64_t>(aDim);
    const uint64_t workspaceFloats = 2ULL * m * k + 2ULL * k * n + 4ULL * m * n;
    const size_t workspaceBytes =
        (static_cast<size_t>(workspaceFloats * sizeof(float)) + 31U) & ~static_cast<size_t>(31U);
    return ChemmRunDeviceOperation(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc,
        aBytes, bBytes, cBytes, workspaceBytes);
}
