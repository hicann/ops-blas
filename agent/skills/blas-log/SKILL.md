---
name: blas-log
description: ops-blas 日志框架使用规范。提供 Host 侧 dlog 日志集成模板、日志配置 API 说明和最佳实践。触发：算子开发中需要集成日志、检视日志使用规范、迁移 printf 到 dlog 时。
---

# ops-blas 日志框架

## 概述

ops-blas 使用 CANN dlog 日志系统，通过 `log/log.h` 提供四级日志宏。用户可通过 `aclblasLoggerConfigure` 配置日志输出目标（总开关/stdout/stderr/文件），日志级别通过环境变量 `ASCEND_GLOBAL_LOG_LEVEL` 控制，算子 Host 代码通过 `OP_LOGD/I/W/E` 输出日志。

## 日志级别

| 级别 | 宏 | 场景 |
|------|-----|------|
| Debug | `OP_LOGD` | Tiling 数据 dump、入口追踪、中间变量 |
| Info | `OP_LOGI` | 设备信息、Kernel 启动参数、成功消息 |
| Warning | `OP_LOGW` | 降级路径、非最优配置、兼容性问题 |
| Error | `OP_LOGE` | 参数校验失败、ACL 调用失败、空指针 |

## 代码层分类

ops-blas 仅有两层代码，日志使用方式不同：

| 代码层 | 头文件 | 日志宏 | 说明 |
|--------|--------|--------|------|
| Host 侧 | `log/log.h` | `OP_LOGD/I/W/E(tag, fmt, ...)` | 参数校验、Tiling 计算、Kernel 启动 |
| Kernel 侧 | 无 | 仅 `printf`（调试用） | **禁止**使用 dlog，仅调试时临时使用 |

## 场景索引

| 场景 | 文档 |
|------|------|
| **Log 语句速查表（推荐首查）** | [log-quickref.md](references/log-quickref.md) |
| Host 侧 6 大 log 模板 | [host-log-templates.md](references/host-log-templates.md) |
| 日志配置 API（3 个公开接口） | [log-config.md](references/log-config.md) |
| 最佳实践 + printf 迁移 | [best-practices.md](references/best-practices.md) |
