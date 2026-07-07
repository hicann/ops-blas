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

## 模板 3：Workspace 容量校验日志

Tiling 通过 const 引用传递（不分配 GM），workspace 由 handle 统一管理（不自行 `aclrtMalloc`）。
若算子所需 workspace 超出 handle 当前容量，输出 Error 日志：

```cpp
size_t workSpaceNeed = ...;  // 算子所需的 workspace 大小
if (workSpaceNeed > GetEffectiveWorkspaceSize(h)) {
    OP_LOGE("aclblasSgemv",
            "workspace not enough: need=%zu, have=%zu",
            workSpaceNeed, GetEffectiveWorkspaceSize(h));
    return ACLBLAS_STATUS_EXECUTION_FAILED;
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
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sgemv_kernel.h"
#include "sgemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

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

static aclblasStatus_t LaunchSgemvKernel(aclblasHandle_t handle, aclblasOperation_t trans,
                                          int m, int n, const float* A, int lda,
                                          const float* x, int incx, float* y, int incy)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasSgemv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t useNumBlocks = std::min(static_cast<uint32_t>(n), coreNum);
    SgemvTilingData tiling = CalcSgemvTilingData(trans, m, n, useNumBlocks);

    OP_LOGD("aclblasSgemv", "tiling: m=%u n=%u numBlocks=%u numThreads=%u",
            tiling.m, tiling.n, useNumBlocks, tiling.numThreads);
    OP_LOGI("aclblasSgemv", "launching kernel with %u blocks", useNumBlocks);

    sgemv_kernel_do(reinterpret_cast<GM_ADDR>(const_cast<float*>(A)),
                    reinterpret_cast<GM_ADDR>(const_cast<float*>(x)),
                    reinterpret_cast<GM_ADDR>(y),
                    reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(h)),
                    useNumBlocks, tiling, h->stream);
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

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t validRet = ValidateSgemvParams(trans, m, n, lda, incx, incy, A, x, y);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    return LaunchSgemvKernel(handle, trans, m, n, A, lda, x, incx, y, incy);
}
```
