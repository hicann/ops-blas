---
name: blas-new-op-workflow
description: ops-blas 算子全流程开发技能，协调 agent 团队完成设计->开发->验收->上库的完整开发链路。触发：用户要求开发新算子或实现某 BLAS 接口时。
---

# 硬规则

以下规则在任何阶段都必须遵守，不可跳过、不可变通。

1. **角色分工** — 主 Agent 只负责调度，禁止亲自动手
   - 每个步骤调用对应 Subagent：编码→`developer`，测试→`tester`，设计→`architect`，检视→`reviewer`，文档→`writer`
   - **禁止**主 Agent 执行：读模板文件（assets/*.md）、读代码、搜索仓内文件、抓取/搜索网页资料、写代码、写文档
   - 主 Agent 唯一例外：读取 references/ 下的流程参考文件、更新 LOG.md、发送 AskUserQuestion
   - 调用 Subagent 时只定义三要素（输入、输出、验收标准），禁止干涉实现细节
   - 若检视/验收报告发现问题，必须通过 developer Subagent 修复，主 Agent 不得直接修改代码

2. **流程控制** — 阶段不可跳过，门控不可绕过
   - 设计→开发→验收→上库，每阶段需用户确认后推进
   - Subagent 报告未通过验证时，禁止进入下一阶段
   - CP1.2 之后的 CP1.4 / CP2.1 / CP2.2：评审/验收通过 → 直接进入下一步（不询问用户）；不通过 → 直接打回修订（不询问用户），同一轮超过 3 次仍不通过再生成 CP 问卷。CP3.2 和 CP4.3 除外，必须询问用户
   - 用户在 CP 问卷中选择「需要修改」时，按以下路径打回：

     | CP | 「需要修改」打回 | 「终止流程」 |
     |----|----------------|------------|
     | CP1.1.A | 重新发送问卷 | — |
     | CP1.1.B | 重新发送问卷 | — |
     | CP1.2 | 打回 1.2 需求分析 | — |
     | CP1.4 | 打回 1.3.A/1.3.B 重新设计 | 终止开发流程 |
     | CP2.1 | 打回 2.1.1.A/2.1.1.B 重新开发 | 终止开发流程 |
     | CP2.2 | 打回 2.2.1.A/2.2.1.B 重新开发 | 终止开发流程 |
     | CP3.2 | 打回 3.1/3.2 重新检视/验收 | — |
     | 4.1.1 / 4.1.2 | 打回 4.1 修复 README | — |
    | CP4.3 | 打回 4.2 代码检视 | — |

3. **问卷处理** — 基于已生成的 json 模板发送 `AskUserQuestion` 问卷时，不得进行任何修改，直接发送。每次问卷得到用户答复后，保存为 `{原问卷名}.ret.json`，仅保留用户选择的选项，删除未选选项。

4. **日志与 Git 安全**
   - 每次流程步骤推进后，主 Agent 必须立即更新 LOG.md（当前步骤、进度状态、开发记录），不得延迟或累积到后续步骤
   - `.agent/` 目录下的所有文件均为临时文件，已被 `.gitignore` 屏蔽。禁止在任何阶段执行 `git add .agent/` 或将该目录下的文件加入版本控制。若发现误加，必须立即 `git reset` 撤销

5. **检视前 diff 预检** — 调用 reviewer（3.1 / 4.2）前，主 Agent 必须先执行以下两个前置步骤：
   - **diff 预检**：执行 `git diff --stat cann/master...HEAD`，将完整输出包含在 reviewer 的 prompt 中。若 diff 中出现非本算子文件（允许的文件白名单：`blas/{operator_name}/`、`test/{operator_name}/`、`include/cann_ops_blas.h`、`test/frame/fill.h`、`test/frame/csv_loader.h`），主 Agent 必须在调用 reviewer 前先让 developer 还原无关变更。
   - **OAT 合规扫描**：执行 `git diff --name-only cann/master...HEAD` 获取变更文件列表，然后执行 `sh scripts/oat_check.sh <变更文件列表>`。若扫描发现问题（License Header Invalid 或 Invalid File Type），必须先让 developer 修复后重新扫描，扫描通过后方可调用 reviewer。扫描报告路径：`oat_reports/result.txt`，须包含在 reviewer 的 prompt 中。

6. **异步 Kernel 启动与 const 引用 Tiling**
   - Host 侧 kernel 是**异步**的：调用 `kernel_do(...)` 后立即返回，**禁止**在 host 侧调用 `aclrtSynchronizeStream`（上层调用方负责同步）
   - Tiling 数据传递方式：host 侧 `TilingData` 结构体以 **`const` 引用**（`const {{Op}}TilingData&`）传入 `kernel_do`，**禁止**使用 `aclrtMalloc` + `aclrtMemcpy(H2D)` + `GM_ADDR tilingGm` 传递 tiling
   - Kernel 侧 `kernel` 函数参数中，tiling 为 **by value**（`const {{Op}}TilingData tiling`），通过运行时 launch 参数自动拷贝，不得再解析 `GM_ADDR tilingGm`
   - 数据 GM 指针仍通过 `uint8_t*`（host 侧）/ `GM_ADDR`（kernel 侧）传递，仅 tiling 参数采用 const 引用
7. **Workspace 由 handle 统一管理**
   - 算子内部**禁止**使用 `aclrtMalloc` 额外分配 workspace。`aclblasCreate` 会预分配 32 MiB 默认 workspace，用户也可通过 `aclblasSetWorkspace` 注入自定义 workspace
   - 需要从 handle 获取当前生效的 workspace：指针用 `GetEffectiveWorkspace(h)`，大小用 `GetEffectiveWorkspaceSize(h)`
   - 若算子所需 workspace 超过默认 32 MiB，应在需求分析文档（1.2）和开发方案设计（1.3.A）中明确说明，由上层调用方在调用前通过 `aclblasSetWorkspace` 注入；**不得**在算子代码中自行扩容
8. **2 层目录结构**
   - 算子代码位于 `blas/{operator_name}/archXX/`（**2 层**，不再使用 `{family}/` 中间层目录）
   - 测试代码位于 `test/{operator_name}/`（同样 2 层）
   - `{operator_name}` 使用 **snake_case** 格式（如 `sswap`、`sgeqrf_batched`、`getri_batched`），**不是** API 名（`aclblasSgeqrfBatched`）
9. **独立 kernel.h 头文件**
   - 每个算子必须有一个独立的 `{operator_name}_kernel.h` 文件，声明 `kernel_do` 函数签名和相关常量
   - host.cpp 和 kernel.cpp 都 `#include "{op}_kernel.h"` 共享该头文件
   - **禁止**在 host.cpp 中以 `extern` 前向声明方式声明 `kernel_do`（也不再使用已删除的公共头文件 `common/kernel_launch/aclblas_kernel_do.h`）
10. **Host 函数结构与强制 dlog 集成**
    - host.cpp 必须拆分为两个静态函数：`Validate{Op}Params(...)` 负责参数校验，`Launch{Op}Kernel(...)` 负责 tiling 计算 + 异步 launch；API 入口函数 `aclblas{OpName}` 只做调度
    - **强制集成 dlog 日志**：host.cpp 必须包含 `#include "log/log.h"`，使用 `OP_LOGE` 记录参数校验失败和 ACL Runtime 失败，使用 `OP_LOGD` 记录 tiling 数据，使用 `OP_LOGI` 记录 kernel launch 信息
    - **禁止**使用 `printf` 或 `std::cout` 输出日志
11. **kernel.h + kernel 签名规范**
    - `kernel.h` 中 `kernel_do` 数据指针参数统一使用 `GM_ADDR`，与 kernel.cpp 签名一致，禁止使用 `uint8_t*`
    - kernel.cpp 中所有 `__global__` kernel 入口函数必须带 `extern "C"` 修饰，禁止 C++ name mangling（reviewer 检视为 HIGH）
12. **host 公共函数复用**
    - host.cpp **禁止**在文件内定义本地 `static GetAivCoreCount` / `GetVectorCoreCount`；必须 `#include "common/helper/host_utils.h"` 使用公共 `GetAivCoreCount()`
    - `GetAivCoreCount` 失败时错误信息统一为 `OP_LOGE("aclblas{Op}", "GetAivCoreCount failed")`，返回 `ACLBLAS_STATUS_INTERNAL_ERROR`（而非 `EXECUTION_FAILED`）
13. **host include 精简**
    - host.cpp **禁止**引入冗余 include：`acl/acl.h`、`cann_ops_blas_common.h`、`tiling/platform/platform_ascendc.h` 均为冗余，由 `host_utils.h` / `aclblas_handle_internal.h` / `kernel.h` 间接引入
    - 仅保留必需头文件：`log/log.h`、`cann_ops_blas.h`、`{op}_kernel.h`、`aclblas_handle_internal.h`、`host_utils.h`；视算子需求可选 `kernel_constant.h`
14. **README 质量门控**
    - README 必须先通过内容审查（4.1.1）和编译测试（4.1.2）才能进入代码检视（4.2）
    - 4.1.1 或 4.1.2 失败时，打回 4.1 writer 修复文档，最多 2 次；第 2 次仍失败通过 AskUserQuestion 询问用户
    - 4.1.2 编译测试中 NPU 不可用时，标记为「跳过运行时（环境限制）」，编译通过即视为成功
    - 调用示例必须使用 RAII 模式（AclContext 类 + std::unique_ptr），对齐 compile_and_run_example.md

---

# 启动指令

**加载本技能后，主 Agent 必须立即执行以下动作，不得做任何其他操作（包括读文件、搜索、询问用户）：**

1. 确定当前步骤（首次加载从 1.1 开始，中断恢复从 LOG.md 记录的步骤继续）
2. 查找下方流程表中该步骤对应的"参与角色"
3. 调用该 Subagent，传入 task-prompts.md 中定义的输入/输出/验收标准
4. Subagent 返回后，主 Agent 更新 LOG.md，然后按流程表推进到下一步

---

# 流程全景

## 角色

| 角色 | 职责 |
|------|------|
| 用户 | 需求提出、各确认点审批 |
| writer | 资料准备、文档与问卷整理、文档编写 |
| architect | 需求分析、方案设计、方案评审 |
| developer | 代码开发、编译联调、性能调优 |
| tester | 测试设计、用例开发、测试验收 |
| reviewer | 代码检视：规范、一致性、风险 |

## 流程

| 步骤 | 输入 | 参与角色 | 输出 | 说明 | 并行 |
|------|------|--------|------|------|------|
| **阶段1：设计** | | | | | |
| 1.1.A 资料准备 | 用户需求 | writer | 工作区目录、LOG.md、1.1-参考资料清单.md | 从用户需求推断临时 operator_name，初始化目录 + 下载资料 | |
| 1.1.S 总结 | 1.1-参考资料清单.md | writer | CP1.1.A.json | 读取参考资料清单，整理为基础信息问卷 | |
| ⛔ CP1.1.A | CP1.1.A.json | 用户 | 算子名/dtype/目标芯片对齐 | AskUserQuestion 对齐基础信息，确认 operator_name | |
| 1.1.B 环境准备 | CP1.1.A确认的算子名 | developer | 2.0.1-开发环境.md、git 分支 | 环境检查、创建分支（使用确认后的 operator_name） | |
| 1.1.S2 总结 | CP1.1.A结论 + 1.1-参考资料清单.md | writer | CP1.1.B.json | 根据 dtype/芯片裁剪，整理接口与参考问卷 | |
| ⛔ CP1.1.B | CP1.1.B.json | 用户 | 精度标准/编程模型对齐 | AskUserQuestion，通过后 git commit -m "CP1.1: 已完成算子基础信息与精度标准对齐" | |
| 1.2 需求分析 | CP1.1结论 + 1.1-参考资料清单.md | architect | 1.2-需求分析.md、CP1.2.json | 参数约束、可行性评估 + 整理为问卷 | |
| ⛔ CP1.2 | CP1.2.json | 用户 | 需求分析审批 | AskUserQuestion，通过后 git commit -m "CP1.2: 已完成需求分析" | |
| 1.3.A 开发方案设计 | 1.2-需求分析.md | architect | 1.3.A-开发方案设计.md | Tiling / Kernel / Host 设计 | ┐ |
| 1.4.A 开发方案评审 | 1.3.A-开发方案设计.md | architect | 1.4.A-开发方案评审.md | 不通过 → 打回 1.3.A，循环 ≤3 次 | │ |
| 1.3.B 测试方案设计 | 1.2-需求分析.md | tester | 1.3.B-测试方案设计.md | 用例表 + 验收标准 | │ |
| 1.4.B 测试方案评审 | 1.3.B-测试方案设计.md、1.2-需求分析.md | tester | 1.4.B-测试方案评审.md | 不通过 → 打回 1.3.B，循环 ≤3 次 | ┘ |
| ⚪ CP1.4 | 1.4.A + 1.4.B | 主Agent | 裁定 | 双方通过→阶段2 + git commit -m "CP1.4: 已完成方案设计"，任一方3次失败→CP问卷 | |
| **阶段2：开发** | | | | | |
| **2.1 迭代一** | | | | 核心路径 | |
| 2.1.1.A 算子开发 | 1.3.A-开发方案设计.md | developer | 算子代码 | 核心逻辑 + 框架 + dlog 日志集成 | ┐ |
| 2.1.1.B 测试开发 | 1.3.B-测试方案设计.md | tester | L0 用例代码 | CSV + golden + GTest | ┘ |
| 2.1.2 汇合联调 | 算子代码 + 用例代码 | developer | 2.1.2-汇合联调报告.md | 编译 + L0 精度 | |
| 2.1.3 测试验收 | 2.1.2-汇合联调报告.md | tester | 2.1.3-测试验收报告.md | L0 通过率 100%，不通过→打回 | |
| ⚪ CP2.1 | 2.1.3-测试验收报告.md | 主Agent | 裁定 | 通过→迭代二 + git commit -m "CP2.1: 已完成迭代一"，不通过→打回 | |
| **2.2 迭代二** | | | | 全覆盖 + 边界 + 异常 | |
| 2.2.1.A 算子开发 | 1.3.A-开发方案设计.md | developer | 完整算子代码 | 补齐分支 + 异常拦截 + 完善日志 | ┐ |
| 2.2.1.B 测试开发 | 1.3.B-测试方案设计.md | tester | L0+L1 用例代码 | 新增 CSV 行 + 测试分支 | ┘ |
| 2.2.2 汇合联调 | 算子代码 + 用例代码 | developer | 2.2.2-汇合联调报告.md | 编译 + 全量精度 | |
| 2.2.3 测试验收 | 2.2.2-汇合联调报告.md | tester | 2.2.3-测试验收报告.md | 全量通过率 100%，不通过→打回 | |
| ⚪ CP2.2 | 2.2.3-测试验收报告.md | 主Agent | 裁定 | 通过→阶段3 + git commit -m "CP2.2: 已完成迭代二"，不通过→打回 | |
| **阶段3：验收** | | | | | |
| 3.1 代码检视 | git diff + OAT checklist + OAT 扫描报告 + 全部变更文件 | reviewer | 3.1-代码检视报告.md | 变更范围、OAT 合规复核、规范、一致性、风险、日志规范；不通过 → developer 修复后重新检视（≤3 次），通过后进入 3.2 | |
| 3.2 性能验收 | 1.2-需求分析.md、1.3.A-开发方案设计.md | developer | 3.2-性能报告.md | 性能采集、瓶颈分析（3.1 检视通过后执行，基于修复后的最终代码） | |
| ⛔ CP3.2 | 3.1 + 3.2 | 用户 | 验收审批 | AskUserQuestion，通过后 git commit -m "CP3.2: 已完成验收" | |
| 3.3 大 shape 精简 | CP3.2 问卷结果 | developer | 精简后的 CSV + ST 通过 | 仅当用户选择「精简为 1 条」时执行 | |
| **阶段4：上库** | | | | | |
| 4.1 编写文档 | 全部代码和设计文档 | writer | README.md | — | |
| 4.1.1 README 内容审查 | README.md + API 声明（cann_ops_blas.h）+ host.cpp | reviewer (scene: readme-review) | 4.1.1-审查报告.md | 9 项逐项审查（模板完整性、API 签名、参数类型、RAII、API 名称、头文件、交叉引用、内存标注、约束），不通过→打回 4.1（≤2 次） | |
| 4.1.2 README 编译测试 | README.md + 2.0.1-开发环境.md | developer (scene: readme-compile-test) | 4.1.2-编译测试报告.md | 提取调用示例 → CMake 编译 → NPU 可用时运行，不通过→打回 4.1 后重跑 4.1.1+4.1.2（≤2 次） | |
| 4.2 代码检视 | git diff + OAT checklist + OAT 扫描报告 + 全部变更文件 + 文档 | reviewer | 4.2-代码检视报告.md | 变更范围 + OAT 合规复核 + 规范 + 冗余清理 + 日志规范 | |
| 4.3 开发总结 | 全部交付物 | writer | CP4.3.json、4.3-Issue.md、4.3-上库PR模板.md、更新 LOG.md | 整理为问卷 + 提 Issue（内容来自需求文档）+ 生成上库 PR 描述 + 更新开发日志 | |
| ⛔ CP4.3 | CP4.3.json | 用户 | 上库审批 | AskUserQuestion，通过后 squash commit -m "Feat: 新增面向archXX的aclblasXxx接口" | |

**图例**：⛔ 必需确认  ⚪ 仅不通过时直接打回，3次仍失败后询问

---

# 参考资源

| 资源 | 路径 | 说明 |
|-----|------|------|
| Task 调用参数 | [references/task-prompts.md](references/task-prompts.md) | 各阶段 Subagent 的调用参数与验收标准 |
| 数据流说明 | [references/data-flow.md](references/data-flow.md) | 各阶段输入输出文件和数据流向 |
| 错误处理指南 | [references/error-handling.md](references/error-handling.md) | 常见错误类型、重试阈值与回退策略 |
| Issue 模板 | [assets/ISSUE_TEMPLATE.md](assets/ISSUE_TEMPLATE.md) | 问题记录模板 |
| 文档与问卷模板 | [assets/](assets/) | 所有产出物的模板文件 |
| README 标准模板 | [assets/README.md](assets/README.md) | 算子 README 统一模板（占位符体系、章节结构、参数表规范），4.1 编写文档时使用。规范原文见 [docs/zh/develop/readme_develop_guide.md](../../../docs/zh/develop/readme_develop_guide.md) |
| ST 测试开发指南 | [docs/zh/develop/st_develop_guide.md](../../../docs/zh/develop/st_develop_guide.md) | GTest + CSV 精度 ST 框架设计与编码规范，2.1.1.B / 2.2.1.B 测试开发时参考 |
| 算子代码模板库 | [agent/skills/blas-op-templates/SKILL.md](../../../agent/skills/blas-op-templates/SKILL.md) | 按编程模型（SIMD membase/regbase/SIMT）分类的标准化代码骨架，2.1.1.A / 2.2.1.A 算子开发时使用 |
