# 最佳实践

## 日志级别选择指南

| 场景 | 级别 | 示例 |
|------|------|------|
| 参数校验失败 | Error | `n <= 0`、空指针、`lda < m` |
| ACL Runtime 调用失败 | Error | `aclrtMalloc`、`aclrtMemcpy` 返回非 `ACL_SUCCESS` |
| Kernel 启动失败 | Error | `aclrtSynchronizeStream` 超时 |
| Tiling 数据 dump | Debug | 完整的 TilingData 结构体字段 |
| 入口/出口追踪 | Debug | 函数入口参数值 |
| 设备信息查询 | Info | 核数、stream 信息 |
| Kernel 启动参数 | Info | block 数、thread 数 |
| 成功完成 | Info | "tiling calculation success" |
| 降级路径 | Warning | 使用 fallback kernel、非最优配置 |
| 兼容性问题 | Warning | 旧版本 API、已废弃参数 |

## 日志消息格式规范

### 标准格式

```cpp
OP_LOGx("算子名", "描述信息, key1=%v1, key2=%v2", val1, val2);
```

- 第一个参数：算子名（字符串），如 `"aclblasSgemv"`
- 第二个参数：描述 + 键值对，使用 printf 格式化
- 键值对用 `key=value` 格式，多个用逗号分隔

### 正确示例

```cpp
OP_LOGE("aclblasSgemv", "m must be positive, got %d", m);
OP_LOGD("aclblasSgemv", "tiling: m=%u n=%u numBlocks=%u", tiling.m, tiling.n, useNumBlocks);
OP_LOGI("aclblasSgemv", "launching kernel with %u blocks on %u cores", useNumBlocks, aivCoreNum);
```

### 错误示例

```cpp
OP_LOGE("aclblasSgemv", "error");
OP_LOGE("", "m is invalid");
OP_LOGE("aclblasSgemv", "m=" + std::to_string(m));
```

## 条件日志（避免性能开销）

当 Debug 日志内容较重时，先检查日志级别：

```cpp
if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
    std::ostringstream ss;
    ss << "full tiling dump: ";
    for (int i = 0; i < 100; ++i) {
        ss << "field" << i << "=" << tiling.getField(i) << " ";
    }
    OP_LOGD("aclblasSgemv", "%s", ss.str().c_str());
}
```

**何时需要条件检查**：
- 字符串拼接（`std::ostringstream`、`std::string` 操作）
- 循环生成日志内容
- 大量字段 dump

**何时不需要**：
- 简单的 printf 格式化（`OP_LOGD("tag", "m=%u", m)`）
- dlog 内部会自动过滤未开启的级别

## printf 迁移

ops-blas 现有代码中使用 `printf` 或 `LOG_PRINT` 的地方，应迁移到 `OP_LOGE`。

### 迁移对照表

| 原代码 | 迁移后 |
|--------|--------|
| `printf("[ERROR][aclblasSgemv] m is invalid\n");` | `OP_LOGE("aclblasSgemv", "m is invalid");` |
| `LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);` | `OP_LOGE("aclblasSgemv", "aclrtMalloc failed, ret=%d", ret);` |
| `printf("tiling: m=%d n=%d\n", m, n);` | `OP_LOGD("aclblasSgemv", "tiling: m=%u n=%u", m, n);` |

### 迁移示例

**迁移前**（`blas/tpttr/stpttr/arch35/stpttr_host.cpp`）：

```cpp
if (n <= 0) {
    printf("[ERROR][aclblasStpttr] n must be positive, got %d\n", n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

**迁移后**：

```cpp
if (n <= 0) {
    OP_LOGE("aclblasStpttr", "n must be positive, got %d", n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

## Kernel 侧注意事项

**Kernel 代码中禁止使用 dlog**。Kernel 运行在 NPU 上，无法访问 host 侧的 dlog 库。

- 调试时可临时使用 `printf`（仅调试用，提交前删除）
- 生产代码中 Kernel 侧不应有任何日志输出

```cpp
__global__ void sgemv_kernel(...) {
    // 错误：Kernel 中不能使用 OP_LOGD
    // OP_LOGD("sgemv_kernel", "blockIdx=%d", blockIdx);

    // 调试时可临时使用 printf（提交前删除）
    // printf("blockIdx=%d\n", blockIdx);

    // 正确：Kernel 中不输出日志
    // ... 计算逻辑 ...
}
```

## 单元测试中的 log stub

在单元测试中，`CheckLogLevel` 需要 stub 以避免依赖 dlog 库：

```cpp
int32_t CheckLogLevel(int32_t module_id, int32_t log_level) {
    return 0;
}
```

- 返回 `0` 表示所有日志级别未开启
- 测试代码中不会执行 Debug 日志分支
- 如需测试 Debug 日志，可返回 `1`

## 常见陷阱

### 1. 忘记包含头文件

```cpp
// 错误：编译报错 OP_LOGE 未定义
OP_LOGE("aclblasSgemv", "error");

// 正确：包含 log/log.h
#include "log/log.h"
OP_LOGE("aclblasSgemv", "error");
```

### 2. 日志级别使用不当

```cpp
// 错误：参数校验失败使用 Info 级别
if (n <= 0) {
    OP_LOGI("aclblasSgemv", "n is invalid");
    return ACLBLAS_STATUS_INVALID_VALUE;
}

// 正确：参数校验失败使用 Error 级别
if (n <= 0) {
    OP_LOGE("aclblasSgemv", "n must be positive, got %d", n);
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

### 3. 日志消息缺少上下文

```cpp
// 错误：缺少参数值
OP_LOGE("aclblasSgemv", "lda is invalid");

// 正确：包含参数值
OP_LOGE("aclblasSgemv", "lda must be >= max(1, m), got lda=%d, m=%d", lda, m);
```

### 4. Debug 日志未做条件检查

```cpp
// 错误：每次都执行字符串拼接，即使 Debug 未开启
std::string msg = buildExpensiveDebugString();
OP_LOGD("aclblasSgemv", "%s", msg.c_str());

// 正确：先检查级别
if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
    std::string msg = buildExpensiveDebugString();
    OP_LOGD("aclblasSgemv", "%s", msg.c_str());
}
```

### 5. Kernel 中使用 dlog

```cpp
// 错误：Kernel 中无法编译
__global__ void kernel() {
    OP_LOGD("kernel", "blockIdx=%d", blockIdx);
}

// 正确：Kernel 中不输出日志
__global__ void kernel() {
    // ... 计算逻辑 ...
}
```
