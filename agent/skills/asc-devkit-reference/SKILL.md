---
name: asc-devkit-reference
description: Ascend C 算子开发工具包参考库。在架构设计、代码开发、API 查阅和性能优化阶段，从 asc-devkit 仓库中检索 API 文档、示例代码、实现参考和构建配置作为开发指导。触发：需要查阅 Ascend C API 官方文档、查找示例代码、参考算子实现、了解构建系统或学习编程模型时。
---

# Ascend C 算子开发工具包参考库

## 概述

本技能管理 [asc-devkit](https://gitcode.com/cann/asc-devkit.git) 仓库的本地副本，为算子开发各阶段提供 Ascend C 官方 API 文档、示例代码、实现参考和构建配置的全面支持。该仓库是 CANN 推出的昇腾 AI 处理器专用的算子程序开发语言，原生支持 C 和 C++ 标准规范，主要由类库和语言扩展层构成，提供多层级 API，满足多维场景算子开发需求。

## 仓库管理

### 初始化

首次使用时，将仓库克隆到 `.agent/asc-devkit`：

```bash
git clone https://gitcode.com/cann/asc-devkit.git .agent/asc-devkit
```

### 更新

若本地副本已存在，拉取最新内容：

```bash
git -C .agent/asc-devkit pull --rebase
```

> 每次使用前应检查是否需要更新，确保参考的是最新文档和示例。

## 仓库结构

```
asc-devkit/
├── docs/                       # 官方文档
│   ├── api/                    # API 文档
│   │   ├── context/            # API 上下文文档（1022+ 个 API 文档，含 figures/ 配图）
│   │   └── README.md           # API 文档索引
│   ├── guide/                  # 开发指南
│   └── README.md               # 文档总入口
├── examples/                   # 示例代码（587+ 个示例）
│   ├── 00_introduction/        # 入门示例：基本编程模型、数据搬运、计算指令
│   ├── 01_utilities/           # 工具类示例：printf 调试、断言使用
│   ├── 02_features/            # 特性示例：C API、SIMT、Micro API、Tiling 等
│   └── 03_libraries/           # 库函数示例：数学库等
├── impl/                       # 实现代码
│   ├── adv_api/                # 高阶 API 实现
│   │   └── tiling/             # Tiling 参数配置参考
│   └── ...                     # 其他实现
├── include/                    # 头文件
│   ├── ascendc/                # Ascend C 核心头文件
│   ├── utils/                  # 工具类头文件
│   └── ...                     # 其他头文件
├── cmake/                      # 构建配置
├── scripts/                    # 脚本工具
├── tests/                      # 单元测试
├── tools/                      # 辅助工具
├── CMakeLists.txt              # 根 CMake 配置
└── README.md                   # 仓库说明
```

> **注意**：以上为仓库的大致分类结构，具体包含哪些文档和子目录会随仓库更新而变化。使用时应直接进入对应目录查看当前可用的内容。

## 各阶段参考指引

### 阶段一：架构设计

在设计 Tiling 策略、Kernel 结构和 Host 流程时，参考仓库中的文档和示例：

1. **API 能力调研** — 查阅 `docs/api/context/` 目录下的 API 文档，了解可用 API 的功能、参数约束和平台支持情况，为设计方案选型提供依据
2. **编程模型选择** — 查看 `examples/02_features/` 中的 SIMT、Micro API 等特性示例，了解不同编程模型的适用场景和架构特点
3. **Tiling 策略设计** — 参考 `impl/adv_api/tiling/` 中的 Tiling 参数配置，了解官方推荐的 Tiling 策略和参数计算方法
4. **同类算子参考** — 在 `examples/` 中查找与目标算子类型相近的示例，参考其整体架构设计思路

**操作方式**：进入 `.agent/asc-devkit/` 对应子目录，阅读 README.md 了解概述，再进入具体目录查看文档和源码。

### 阶段二：代码开发

在编写 Host/Kernel/Tiling 代码时，参考仓库中的 API 文档和示例代码：

1. **API 用法查阅** — 查阅 `docs/api/context/` 中目标 API 的官方文档，确认函数签名、参数类型、约束条件和平台支持。**注意**：同一 API 可能有多个变体文件（如 `Add.md` / `Add-25.md`），必须用通配符搜索所有变体并逐一查阅
2. **API 配图细读** — `docs/api/context/figures/` 下的配图（.png/.jpg/.svg）常承载文字未明确表达的关键约束（流水时序、内存布局、参数示意等），必须使用 Read 工具逐张查看
3. **示例代码参考** — 参考 `examples/00_introduction/` 中的入门示例，学习 Ascend C API 的标准调用方式（数据搬运、计算指令、同步 barrier 等）
4. **特性代码参考** — 查看 `examples/02_features/` 中的特性示例，学习 SIMT Kernel、C API、Micro API 等特定编程模型的代码写法
5. **头文件查阅** — 查阅 `include/ascendc/` 中的头文件，了解类型定义、模板参数和宏定义
6. **构建配置** — 参考 `cmake/` 和 `CMakeLists.txt` 了解编译配置和依赖管理方式

**操作方式**：直接阅读 `.agent/asc-devkit/` 下目标文件。API 文档为 Markdown 格式，示例代码为 `.asc`/`.cpp`/`.h` 格式。

### 阶段三：性能优化

在进行性能调优和瓶颈分析时，参考仓库中的优化示例和工具：

1. **性能优化示例** — 查看 `examples/` 中与性能相关的示例，学习 Double Buffer、流水线并行、数据搬运优化等编程模式
2. **Tiling 参数调优** — 参考 `impl/adv_api/tiling/` 中的 Tiling 配置，了解不同参数对性能的影响
3. **工具使用** — 参考 `tools/` 和 `scripts/` 中的辅助工具，学习 Profiling 和仿真分析方法
4. **单元测试参考** — 查看 `tests/` 中的单元测试，了解 API 的正确使用方式和边界条件处理

**操作方式**：优先阅读相关示例目录中的 README.md，理解优化思路，再对照代码实现学习具体优化手法。

## API 文档检索指南

### 变体搜索（重要）

Ascend C 存在 **240+ 个带数字后缀的 API 变体**（如 `Add-25.md`），同名 API 的不同变体功能可能完全不同。

#### 强制搜索步骤

1. **列出所有变体**：
   ```bash
   ls .agent/asc-devkit/docs/api/context/ | grep -iE "^APIName"
   ```

2. **逐一确认功能**：每个变体的函数签名、参数、功能可能完全不同

#### 变体命名规律

| 后缀 | 含义 | 示例 |
|------|------|------|
| 无后缀 | 基础版本 | `Add.md` |
| `-数字` | 变体版本（数字无语义，功能可能完全不同） | `Add-25.md` |

#### 变体检测命令

```bash
# 查找某个 API 的所有变体（强制）
ls .agent/asc-devkit/docs/api/context/ | grep -iE "^APIName"

# 在所有变体中搜索特定关键词
grep -l "关键词" .agent/asc-devkit/docs/api/context/APIName*.md
```

### 配图查阅

API 文档中引用的配图存放在 `docs/api/context/figures/` 目录下，包含：
- **公式图**：确认数学语义
- **流水时序图**：理解 MTE2/V/MTE3 的依赖与并行关系
- **内存布局图**：UB 槽位摆放规则、对齐边界
- **参数示意图**：stride / block 在 UB/GM 的几何含义

> **强制要求**：含配图（`figures/*.png/jpg/svg`）的 API 文档，必须使用 Read 工具逐张查看配图，禁止仅看正文文字。

## 检索策略

当需要查找参考时，按以下顺序检索：

1. **先读文档索引** — 从 `docs/README.md` 和 `docs/api/README.md` 开始，定位相关文档
2. **再读 API 文档** — 进入 `docs/api/context/` 查找目标 API 的所有变体文档
3. **查阅示例代码** — 进入 `examples/` 对应子目录，查看 README.md 和源码文件
4. **查阅实现代码** — 进入 `impl/` 查看官方实现参考
5. **查阅头文件** — 进入 `include/` 查看类型定义和接口声明

> 仓库内容持续更新，若在某个分类下未找到相关参考，应检查仓库是否有新增目录或文件。也可查看仓库根目录 README.md 了解最新变更。

## 与其他技能的关系

| 技能 | 职责分工 | 协作方式 |
|------|---------|---------|
| `ascendc-docs-search` | 在线文档搜索（华为昇腾社区） | 本地 asc-devkit 文档不足时，使用 ascendc-docs-search 在线搜索兜底 |
| `ascendc-api-best-practices` | API 使用约束和最佳实践 | 查阅 asc-devkit API 文档后，结合 best-practices 确认使用约束 |
| `op-samples-reference` | cann-samples 高性能样例 | asc-devkit 侧重 API/示例/实现参考，cann-samples 侧重端到端性能调优实践 |

## 注意事项

- 仓库中的示例可能依赖特定版本的 CANN Toolkit 和 NPU 架构，参考时注意查看示例 README 中的环境要求
- 示例代码的风格和规范可能与 ops-blas 项目不完全一致，参考时应以 ops-blas 自身的编码规范（`blas-ascendc-coding-rules`）为准
- API 文档中的参数约束和平台支持信息以官方文档为准，设计方案中引用的 API 必须经过完整验证
