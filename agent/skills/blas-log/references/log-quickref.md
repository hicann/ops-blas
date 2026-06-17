# Log 语句速查表

ops-blas host.cpp 中所有日志语句的统一模板。算子名统一用 `"aclblasXxx"` 格式。

## 前置条件

```cpp
#include "log/log.h"
```

---

## 1. Handle 空指针

```cpp
if (handle == nullptr) {
    OP_LOGE("aclblasXxx", "handle is nullptr");
    return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
}
```

---

## 2. 数值范围校验

### 2.1 维度参数 < 0

```cpp
if (n < 0) {
    OP_LOGE("aclblasXxx", "n must be >= 0, got %d", n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

多参数合并：

```cpp
if (m < 0 || n < 0) {
    OP_LOGE("aclblasXxx", "m and n must be >= 0, got m=%d, n=%d", m, n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 2.2 带宽参数 < 0（kl, ku, k）

```cpp
if (kl < 0) {
    OP_LOGE("aclblasXxx", "kl must be >= 0, got %d", kl);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (ku < 0) {
    OP_LOGE("aclblasXxx", "ku must be >= 0, got %d", ku);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (k < 0) {
    OP_LOGE("aclblasXxx", "k must be >= 0, got %d", k);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 2.3 步长 incx/incy 不可为零

```cpp
if (incx == 0) {
    OP_LOGE("aclblasXxx", "incx must not be zero");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (incy == 0) {
    OP_LOGE("aclblasXxx", "incy must not be zero");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

---

## 3. 枚举值校验

### 3.1 uplo（FillMode）

```cpp
if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
    OP_LOGE("aclblasXxx", "uplo must be UPPER(121) or LOWER(122), got %d",
            static_cast<int>(uplo));
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 3.2 trans（Operation）

```cpp
if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
    OP_LOGE("aclblasXxx", "trans must be OP_N(111), OP_T(112) or OP_C(113), got %d",
            static_cast<int>(trans));
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 3.3 diag（DiagType）

```cpp
if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
    OP_LOGE("aclblasXxx", "diag must be NON_UNIT(131) or UNIT(132), got %d",
            static_cast<int>(diag));
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

---

## 4. 前导维度 lda

### 4.1 通用方阵：lda >= max(1, n)

```cpp
if (lda < std::max(1, n)) {
    OP_LOGE("aclblasXxx", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 4.2 带状矩阵：lda >= kl + ku + 1

```cpp
if (lda < kl + ku + 1) {
    OP_LOGE("aclblasXxx", "lda must be >= kl+ku+1, got lda=%d, kl=%d, ku=%d", lda, kl, ku);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 4.3 对称带状：lda >= k + 1

```cpp
if (lda < k + 1) {
    OP_LOGE("aclblasXxx", "lda must be >= k+1, got lda=%d, k=%d", lda, k);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

---

## 5. 数据指针空检查

### 5.1 矩阵/向量指针

```cpp
if (A == nullptr) {
    OP_LOGE("aclblasXxx", "A must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (x == nullptr) {
    OP_LOGE("aclblasXxx", "x must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (y == nullptr) {
    OP_LOGE("aclblasXxx", "y must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

多指针合并：

```cpp
if (A == nullptr || x == nullptr || y == nullptr) {
    OP_LOGE("aclblasXxx", "A/x/y must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 5.2 标量指针（alpha, beta, result）

```cpp
if (alpha == nullptr) {
    OP_LOGE("aclblasXxx", "alpha must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (beta == nullptr) {
    OP_LOGE("aclblasXxx", "beta must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
if (result == nullptr) {
    OP_LOGE("aclblasXxx", "result must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 5.3 压缩存储指针（AP）

```cpp
if (AP == nullptr) {
    OP_LOGE("aclblasXxx", "AP must not be nullptr");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

---

## 6. 设备查询

### 6.1 aclrtGetDevice

```cpp
int32_t deviceId = 0;
aclError aclRet = aclrtGetDevice(&deviceId);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasXxx", "aclrtGetDevice failed, ret=%d", aclRet);
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

### 6.2 aclrtGetDeviceInfo

```cpp
uint32_t aivCoreNum = 0;
aclError aclRet = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aivCoreNum);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasXxx", "aclrtGetDeviceInfo failed, ret=%d", aclRet);
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

### 6.3 核数为零

```cpp
if (aivCoreNum == 0) {
    OP_LOGE("aclblasXxx", "vector core count is 0");
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

---

## 7. Workspace 管理（handle 统一）

> **注意**：Tiling 数据不再通过 `aclrtMalloc`/`aclrtMemcpy(H2D)` 传递（改用 `const TilingData&` 引用直接传给 `kernel_do`，运行时 launch 参数自动拷贝），workspace 由 `aclblasCreate` 预分配 4 MiB 默认容量，也可由用户通过 `aclblasSetWorkspace` 注入。**算子内禁止再 `aclrtMalloc` 分配 tilingDevice 或 workspace**。

### 7.1 Workspace 容量校验（必须）

```cpp
size_t workSpaceNeed = /* 算子所需 workspace 大小 */;
if (workSpaceNeed > aclblasGetEffectiveWorkspaceSize(h)) {
    OP_LOGE("aclblasXxx",
            "workspace not enough: need=%zu, have=%zu",
            workSpaceNeed, aclblasGetEffectiveWorkspaceSize(h));
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

### 7.2 获取当前生效 workspace 指针

```cpp
void* workSpace = aclblasGetEffectiveWorkspace(h);
```

### 7.3 aclrtMemcpy（Device → Host）

```cpp
aclError aclRet = aclrtMemcpy(hostBuf, size, deviceBuf, size, ACL_MEMCPY_DEVICE_TO_HOST);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasXxx", "aclrtMemcpy D2H failed, ret=%d", aclRet);
    aclrtFree(deviceBuf);
    return ACLBLAS_STATUS_INTERNAL_ERROR;
}
```

### 7.4 aclrtMemset

```cpp
aclError aclRet = aclrtMemset(deviceBuf, size, 0, size);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasXxx", "aclrtMemset failed, ret=%d", aclRet);
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

### 7.5 aclrtSynchronizeStream（已禁止在算子内使用）

> **算子 host 侧 launch kernel 后必须异步返回，禁止调用 `aclrtSynchronizeStream`**。同步由上层调用方负责。
> 仅在极少数特殊场景（如上层要求算子自行同步以读取 workspace 中间结果）使用：

```cpp
aclError aclRet = aclrtSynchronizeStream(h->stream);
if (aclRet != ACL_SUCCESS) {
    OP_LOGE("aclblasXxx", "aclrtSynchronizeStream failed, ret=%d", aclRet);
    return ACLBLAS_STATUS_EXECUTION_FAILED;
}
```

---

## 8. Tiling 计算与 Kernel 启动

### 8.1 Tiling 数据 Debug dump（简单）

```cpp
OP_LOGD("aclblasXxx", "tiling: n=%u numBlocks=%u numThreads=%u",
        tiling.n, useNumBlocks, tiling.numThreads);
```

### 8.2 Tiling 数据 Debug dump（完整，需条件检查）

```cpp
if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
    OP_LOGD("aclblasXxx", "tiling: n=%u k=%u lda=%u uplo=%u trans=%u diag=%u incx=%ld "
            "numBlocks=%u numThreads=%u useUb=%u",
            tiling.n, tiling.k, tiling.lda, tiling.uplo, tiling.trans, tiling.diag,
            tiling.incx, useNumBlocks, tiling.numThreads, tiling.useUb);
}
```

### 8.3 Kernel 启动 Info

```cpp
OP_LOGI("aclblasXxx", "launching kernel: blocks=%u, cores=%u", useNumBlocks, aivCoreNum);
```

---

## 9. 早退（不需要日志）

早退是正常逻辑分支，**不输出日志**：

```cpp
if (n == 0) {
    return ACLBLAS_STATUS_SUCCESS;
}
if (m == 0 || n == 0) {
    return ACLBLAS_STATUS_SUCCESS;
}
```

---

## 10. 完整 ValidateParams 函数模板

```cpp
static aclblasStatus_t ValidateXxxParams(aclblasFillMode_t uplo, int n, int lda,
                                         const float* A, float* x, int incx)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasXxx", "uplo must be UPPER(121) or LOWER(122), got %d",
                static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasXxx", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasXxx", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasXxx", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr || x == nullptr) {
        OP_LOGE("aclblasXxx", "A/x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
```

---

## 统一规范

| 项目 | 规范 |
|------|------|
| tag 格式 | `"aclblasXxx"`（完整 API 名） |
| 错误级别 | 参数校验 / ACL 失败 → `OP_LOGE` |
| 信息级别 | Kernel 启动参数 → `OP_LOGI` |
| 调试级别 | Tiling dump → `OP_LOGD` |
| 返回值统一 | 参数错误 → `INVALID_VALUE`，分配失败 → `ALLOC_FAILED`，ACL 同步/执行失败 → `EXECUTION_FAILED`，ACL 拷贝失败 → `INTERNAL_ERROR`，handle 空 → `HANDLE_IS_NULLPTR` |
| 消息格式 | `"描述, key1=%v1, key2=%v2"` 键值对 |
| 早退 | `n==0` 返回 SUCCESS，**不输出日志** |
| aclrtFree | 不检查返回值（fire-and-forget） |
