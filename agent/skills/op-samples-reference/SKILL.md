---
name: op-samples-reference
description: CANN 算子高性能样例参考库。在架构设计、代码开发和性能优化阶段，从 cann-samples 仓库中检索相关样例代码与调优实践作为参考。触发：需要参考已有算子实现、查找优化方法示例、或学习特定编程模型时。
---

# CANN 算子样例参考库

## 概述

本技能管理 [cann-samples](https://gitcode.com/cann/cann-samples.git) 仓库的本地副本，为算子开发各阶段提供高性能实战样例和调优知识库的参考。该仓库持续更新，包含算子领域的最佳实践、优化方法和完整代码示例。

## 仓库管理

### 初始化

首次使用时，将仓库克隆到 `.agent/cann-samples`：

```bash
git clone https://gitcode.com/cann/cann-samples.git .agent/cann-samples
```

### 更新

若本地副本已存在，拉取最新内容：

```bash
git -C .agent/cann-samples pull --rebase
```

> 每次使用前应检查是否需要更新，确保参考的是最新样例。

## 仓库结构

```
cann-samples/
├── Samples/
│   ├── 0_Introduction/       # 入门样例：NPU 执行链路、Vector/Cube 基本编程模型
│   ├── 1_Features/           # 特性样例：解耦底层优化能力
│   │   ├── memory_optimization/      # 访存优化（带宽利用、缓存策略、bank 冲突等）
│   │   ├── instruction_optimization/ # 指令优化（流水并行、多缓冲、权重格式等）
│   │   ├── system_optimization/      # 系统优化（负载均衡、多核调度等）
│   │   └── hardware_features/        # 芯片特性（SIMT、Vector Function、量化类型等）
│   ├── 2_Performance/        # 性能实战：从 Baseline 到极致性能的端到端调优
│   └── 3_Utilities/          # 工具类：仿真、Profiling 等辅助工具样例
├── third_party/              # 外部依赖（Git 子模块）
├── cmake/                    # 编译配置
└── CMakeLists.txt            # 根 CMake 配置
```

> **注意**：以上为仓库的大致分类结构，具体包含哪些样例和子目录会随仓库更新而变化。使用时应直接进入对应目录查看当前可用的内容。

## 各阶段参考指引

### 阶段一：架构设计

在设计 Tiling 策略、Kernel 结构和 Host 流程时，参考样例中的架构模式：

1. **入门编程模型** — 查看 `Samples/0_Introduction/` 下的样例，了解 Vector Core 和 Cube Core 的基本 Tiling 策略、数据搬运模式和流水并行结构
2. **同类算子实现** — 在 `Samples/2_Performance/` 中查找与目标算子类型相近的样例（如矩阵乘类、归约类、通信类），参考其整体架构设计思路
3. **优化策略选型** — 浏览 `Samples/1_Features/` 各子目录的 README，了解可用的优化手段，在设计方案中提前规划可采用的优化策略
4. **SIMT 算子设计** — 若目标算子采用 SIMT 编程模型，必须参考 `Samples/1_Features/hardware_features/simt/` 中的样例，了解 SIMT 架构下的 Tiling 策略、线程映射和 Kernel 结构设计

**操作方式**：进入 `.agent/cann-samples/Samples/` 对应子目录，阅读 README.md 了解样例概述，再进入具体样例目录查看设计文档和源码。

### 阶段二：代码开发

在编写 Host/Kernel/Tiling 代码时，参考样例中的具体实现：

1. **API 用法** — 参考入门样例中的 Ascend C API 调用方式（数据搬运、计算指令、同步 barrier 等）
2. **Kernel 结构** — 参考性能样例中的 Kernel 实现，学习多缓冲（N-Buffer）、流水并行、分块策略等编程模式
3. **芯片特性利用** — 查看 `Samples/1_Features/hardware_features/` 中的样例，了解 Vector Function 等特定编程模型的代码写法
4. **SIMT 算子开发** — 若目标算子采用 SIMT 编程模型，必须参考 `Samples/1_Features/hardware_features/simt/` 中的样例代码，学习 SIMT Kernel 的具体实现方式、线程索引计算、共享内存使用等
5. **构建配置** — 参考样例中的 CMakeLists.txt 了解编译配置和依赖管理方式

**操作方式**：直接阅读 `.agent/cann-samples/Samples/` 下目标样例的源码文件（`.cpp`/`.h`），关注 kernel 实现、tiling 数据结构和 host 侧调用逻辑。

### 阶段三：性能优化

在进行性能调优和瓶颈分析时，参考样例中的调优实践：

1. **端到端调优路径** — `Samples/2_Performance/` 下的 story 类样例通常包含从 baseline 到极限性能的完整调优过程，包括性能分析文档、分步教程和可运行的 recipe 示例
2. **访存瓶颈** — 参考 `Samples/1_Features/memory_optimization/` 中的样例，学习带宽利用率优化、L1 缓存策略、bank 冲突消除等方法
3. **指令效率** — 参考 `Samples/1_Features/instruction_optimization/` 中的样例，学习流水并行度提升、指令预取、权重格式转换等方法
4. **系统级优化** — 参考 `Samples/1_Features/system_optimization/` 中的样例，学习多核负载均衡、尾轮优化等策略
5. **性能分析工具** — 参考 `Samples/3_Utilities/` 中的工具样例，学习 Profiling 和仿真分析方法

**操作方式**：优先阅读 `Samples/2_Performance/` 下相关 story 目录中的 docs/ 文档和 README.md，理解调优思路，再对照代码实现学习具体优化手法。

## 检索策略

当需要查找参考时，按以下顺序检索：

1. **先读目录索引** — 从 `Samples/` 根目录开始，逐级阅读 README.md 定位相关样例
2. **再读样例文档** — 进入目标样例目录，阅读其 README.md 和 docs/ 下的文档
3. **最后读源码** — 根据文档指引查看具体的 `.cpp`/`.h` 源码文件

> 仓库内容持续更新，若在某个分类下未找到相关参考，应检查仓库是否有新增目录或样例。也可查看仓库根目录 README.md 的 Latest News 了解最近新增的内容。

## 注意事项

- 仓库中的样例可能依赖特定版本的 CANN Toolkit 和 NPU 架构，参考时注意查看样例 README 中的环境要求
- 部分样例使用 `third_party/tensor_api` 子模块提供的 Tensor API，参考时需确认子模块已初始化
- 样例代码的风格和规范可能与 ops-blas 项目不完全一致，参考时应以 ops-blas 自身的编码规范（`blas-ascendc-coding-rules`）为准
