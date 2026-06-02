# 日志配置 API

## 公开接口

ops-blas 提供 3 个日志配置接口，声明在 `include/cann_ops_blas.h`。

### aclblasLoggerConfigure

配置日志输出目标和日志级别。

```cpp
aclblasStatus_t aclblasLoggerConfigure(
    const char* logFile,
    bool logToStdOut,
    bool logToKdlls,
    aclblasLogLevel_t logLevel);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| logFile | `const char*` | 日志文件路径，`nullptr` 表示不输出到文件 |
| logToStdOut | `bool` | 是否输出到标准输出 |
| logToKdlls | `bool` | 是否输出到内核日志 |
| logLevel | `aclblasLogLevel_t` | 日志级别 |

**返回值**：`ACLBLAS_STATUS_SUCCESS` 成功

**调用示例**：

```cpp
aclblasLoggerConfigure("ops_blas.log", true, false, ACLBLAS_LOG_LEVEL_DEBUG);
```

### aclblasSetLoggerCallback

设置用户自定义日志回调函数。

```cpp
aclblasStatus_t aclblasSetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | `aclblasHandle` | ops-blas handle |
| userCallback | `aclblasLogCallback` | 回调函数，签名 `void (*)(char*)` |

**返回值**：`ACLBLAS_STATUS_SUCCESS` 成功，`ACLBLAS_STATUS_HANDLE_IS_NULLPTR` handle 为空

### aclblasGetLoggerCallback

获取当前日志回调函数。

```cpp
aclblasStatus_t aclblasGetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback);
```

## 日志级别枚举

定义在 `include/cann_ops_blas_common.h`：

```cpp
typedef enum aclblasLogLevel {
    ACLBLAS_LOG_LEVEL_DEBUG = 0,
    ACLBLAS_LOG_LEVEL_INFO = 1,
    ACLBLAS_LOG_LEVEL_ERROR = 2,
} aclblasLogLevel_t;
```

| 级别 | 值 | 说明 |
|------|-----|------|
| `ACLBLAS_LOG_LEVEL_DEBUG` | 0 | 最详细，包含 Tiling dump、入口追踪 |
| `ACLBLAS_LOG_LEVEL_INFO` | 1 | 设备信息、Kernel 启动参数 |
| `ACLBLAS_LOG_LEVEL_ERROR` | 2 | 仅错误信息 |

## 底层实现

`aclblasLoggerConfigure` 内部通过 `dlog_setlevel` 将日志级别传递到底层 dlog 系统：

```cpp
switch (logLevel) {
    case ACLBLAS_LOG_LEVEL_INFO:
        dlog_setlevel(OP, DLOG_INFO, 1);
        break;
    case ACLBLAS_LOG_LEVEL_ERROR:
        dlog_setlevel(OP, DLOG_ERROR, 1);
        break;
    case ACLBLAS_LOG_LEVEL_DEBUG:
        dlog_setlevel(OP, DLOG_DEBUG, 1);
        break;
}
```

- `OP` 是 dlog 模块标识符（算子模块）
- 第三个参数 `1` 是设备 ID

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
