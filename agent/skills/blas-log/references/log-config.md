# 日志配置 API

## 公开接口

ops-blas 提供 3 个日志配置接口，声明在 `include/cann_ops_blas.h`。

### aclblasLoggerConfigure

配置日志输出目标和总开关。

```cpp
aclblasStatus_t aclblasLoggerConfigure(
    int logIsOn,
    int logToStdOut,
    int logToStdErr,
    const char* logFile);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| logIsOn | `int` | 是否开启日志（总开关），非零为开启、0 为关闭 |
| logToStdOut | `int` | 是否输出到标准输出，非零为是、0 为否 |
| logToStdErr | `int` | 是否输出到标准错误，非零为是、0 为否 |
| logFile | `const char*` | 日志文件路径，`nullptr` 表示不输出到文件。内部不拷贝，调用方需保证指针在日志输出期间有效 |

**返回值**：`ACLBLAS_STATUS_SUCCESS` 成功

**调用示例**：

```cpp
aclblasLoggerConfigure(1, 1, 0, "ops_blas.log");
```

### aclblasSetLoggerCallback

设置用户自定义日志回调函数。传入 `nullptr` 可清除已安装的回调。

```cpp
aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| userCallback | `aclblasLogCallback` | 回调函数，签名 `void (*)(const char*)`；`nullptr` 表示清除回调 |

**返回值**：`ACLBLAS_STATUS_SUCCESS` 成功

### aclblasGetLoggerCallback

获取当前日志回调函数。

```cpp
aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback);
```

| 返回值 | 含义 |
|------|------|
| `ACLBLAS_STATUS_SUCCESS` | 获取成功 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `userCallback` 为空指针 |

## 日志级别控制

`aclblasLoggerConfigure` 不包含日志级别参数（对齐业界主流 BLAS 库设计）。日志级别通过环境变量 `ASCEND_GLOBAL_LOG_LEVEL` 控制，详见下方"环境变量配置"。

## 底层实现

`aclblasLoggerConfigure` 将配置存储到内部全局结构体，供日志输出逻辑消费。日志级别由 dlog 全局配置（环境变量）控制，与业界主流 BLAS 库通过环境变量控制级别的方式一致。

## 环境变量配置

除 API 外，还可通过环境变量控制日志行为：

| 环境变量 | 说明 | 值 |
|---------|------|-----|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否输出到标准输出 | `"1"` 开启，`"0"` 关闭 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 全局日志级别 | `"0"` = DEBUG，`"1"` = INFO，`"2"` = ERROR |

**使用示例**：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
```

## 运行时日志级别检查

在算子代码中，可通过 `CheckLogLevel` 检查当前日志级别是否开启，避免不必要的性能开销：

```cpp
#include "log/log.h"

if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
    OP_LOGD("aclblasSgemv", "expensive debug info: %s", buildDebugString().c_str());
}
```

- `CheckLogLevel` 返回 `1` 表示该级别已开启
- 仅当 Debug 日志内容较重（如字符串拼接）时才需要检查
- 简单的 `OP_LOGD` 调用无需检查，dlog 内部会自动过滤
