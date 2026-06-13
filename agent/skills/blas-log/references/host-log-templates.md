# Host 侧日志集成模板

ops-blas 的 host.cpp 遵循标准流程，以下提供 6 个精准日志模板。

## 前置条件

```cpp
#include "log/log.h"
```

## 模板 1：Handle 空指针检查

在 `aclblasXxx()` 入口处，handle 为空时输出 Error 日志。

```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, ...) {
    if (handle == nullptr) {
        OP_LOGE("aclblasSgemv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // ...
}
```

## 模板 2：参数校验错误日志

替代现有的静默 `CHECK_RET`，在参数校验失败时输出 Error 日志。

```cpp
static aclblasStatus_t ValidateSgemvParams(aclblasOperation_t trans, int m, int n,
                                           int lda, int64_t incx, int64_t incy,
                                           const float* a, const float* x, float* y)
{
    if (m <= 0) {
        OP_LOGE("aclblasSgemv", "m must be positive, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        OP_LOGE("aclblasSgemv", "n must be positive, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasSgemv", "lda must be >= max(1, m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE("aclblasSgemv", "a/x/y pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
```

## 模板 3：ACL Runtime 调用失败日志

ACL Runtime API（`aclrtMalloc`、`aclrtMemcpy`、`aclrtSynchronizeStream`）失败时输出 Error 日志。

```cpp
aclErrorNumber aclRet = aclrtMalloc(&tilingDevice, sizeof(SgemvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasSgemv", "aclrtMalloc failed, size=%zu, ret=%d", sizeof(SgemvTilingData), aclRet);
    return ACLBLAS_STATUS_ALLOC_FAILED;
}

aclRet = aclrtMemcpy(tilingDevice, sizeof(SgemvTilingData), &tiling, sizeof(SgemvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasSgemv", "aclrtMemcpy failed, ret=%d", aclRet);
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_MAPPING_ERROR;
}
```

## 模板 4：Tiling 数据 Debug dump

计算完 Tiling 数据后，输出 Debug 级别日志，便于调试。

```cpp
OP_LOGD("aclblasSgemv", "tiling: m=%u n=%u numBlocks=%u numThreads=%u rowsPerBlock=%u",
        tiling.m, tiling.n, useNumBlocks, tiling.numThreads, tiling.rowsPerBlock);
```

## 模板 5：Kernel 启动 Info 日志

Kernel 启动前输出 Info 日志，记录关键启动参数。

```cpp
OP_LOGI("aclblasSgemv", "launching kernel with %u blocks on %u vector cores",
        useNumBlocks, aivCoreNum);
```

## 模板 6：条件 Debug 日志（避免性能开销）

当 Debug 日志内容较重（如完整 Tiling dump）时，先检查日志级别是否开启。

```cpp
if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
    std::ostringstream ss;
    ss << "full tiling: ";
    ss << "trans=" << tiling.trans << " alpha=" << tiling.alpha << " beta=" << tiling.beta;
    ss << " incx=" << tiling.incx << " incy=" << tiling.incy;
    OP_LOGD("aclblasSgemv", "%s", ss.str().c_str());
}
```

## 完整 host.cpp 示例

```cpp
#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void sgemv_kernel_do(uint8_t* A, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateSgemvParams(aclblasOperation_t trans, int m, int n,
                                           int lda, int64_t incx, int64_t incy,
                                           const float* a, const float* x, float* y)
{
    if (m <= 0) {
        OP_LOGE("aclblasSgemv", "m must be positive, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        OP_LOGE("aclblasSgemv", "n must be positive, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE("aclblasSgemv", "a/x/y pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans,
                             int m, int n, const float* alpha, const float* A,
                             int lda, const float* x, int incx,
                             const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSgemv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);
    aclblasStatus_t validRet = ValidateSgemvParams(trans, m, n, lda, incx, incy, A, x, y);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = 0;
    // ... GetVectorCoreCount ...

    SgemvTilingData tiling{};
    // ... CalcSgemvTiling ...

    OP_LOGD("aclblasSgemv", "tiling: m=%u n=%u numBlocks=%u numThreads=%u",
            tiling.m, tiling.n, useNumBlocks, tiling.numThreads);

    void* tilingDevice = nullptr;
    aclErrorNumber aclRet = aclrtMalloc(&tilingDevice, sizeof(SgemvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSgemv", "aclrtMalloc failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(tilingDevice, sizeof(SgemvTilingData), &tiling,
                         sizeof(SgemvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSgemv", "aclrtMemcpy failed, ret=%d", aclRet);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_MAPPING_ERROR;
    }

    OP_LOGI("aclblasSgemv", "launching kernel with %u blocks", useNumBlocks);
    sgemv_kernel_do(A, x, y, h->workspace, static_cast<GM_ADDR>(tilingDevice),
                    useNumBlocks, h->stream);

    aclrtSynchronizeStream(h->stream);
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}
```
