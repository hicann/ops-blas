# AclBlas Agent

多 Agent 协作框架，将 BLAS 算子开发流程编排为可追溯、可恢复的工程流水线。用户描述算子需求，Agent 团队自动完成从需求分析到代码上库的全流程。

## 设计思想

**结构化问卷，无遗漏需求** — 逐项问透，不留盲区。

**执行不验收，验收不执行** — 质量靠制衡，不靠自觉。

**流程可追溯，文档全记录** — 每步有日志，每阶段有产出。

## 参与角色

| 角色 | 职责 |
|------|------|
| 用户 | 需求提出、各确认点审批 |
| writer | 资料准备、文档与问卷整理、文档编写 |
| architect | 需求分析、方案设计、方案评审 |
| developer | 代码开发、编译联调、性能调优 |
| tester | 测试设计、用例开发、测试验收 |
| reviewer | 代码检视：规范、一致性、风险 |

## 开发流程

| 步骤 | 输入 | 参与角色 | 输出 | 说明 | 并行 |
|------|------|--------|------|------|------|
| **阶段1：设计** | | | | | |
| 1.1.A 资料准备 | 用户需求 | writer | 工作区目录、LOG.md、1.1-参考资料清单.md | 从用户需求推断临时 operator_name，初始化目录 + 下载资料 | |
| 1.1.S 总结 | 1.1-参考资料清单.md | writer | CP1.1.A.json | 读取参考资料清单，整理为基础信息问卷 | |
| ⛔ CP1.1.A | CP1.1.A.json | 用户 | 算子名/dtype/目标芯片对齐 | AskUserQuestion 对齐基础信息，确认 operator_name | |
| 1.1.B 环境准备 | CP1.1.A确认的算子名 | developer | 2.0.1-开发环境.md、git 分支 | 环境检查、创建分支（使用确认后的 operator_name） | |
| 1.1.S2 总结 | CP1.1.A结论 + 1.1-参考资料清单.md | writer | CP1.1.B.json | 根据 dtype/芯片裁剪，整理接口与参考问卷 | |
| ⛔ CP1.1.B | CP1.1.B.json | 用户 | 精度标准/编程模型对齐 | AskUserQuestion，通过后 git commit -m "CP1.1: 已对齐算子基础信息" | |
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
| 3.1 代码检视 | git diff + OAT checklist + 全部变更文件 | reviewer | 3.1-代码检视报告.md | 变更范围、OAT 合规复核、规范、一致性、风险、日志规范 | |
| 3.2 性能验收 | 1.2-需求分析.md、1.3.A-开发方案设计.md | developer | 3.2-性能报告.md | 性能采集、瓶颈分析 | |
| ⛔ CP3.2 | 3.1 + 3.2 | 用户 | 验收审批 | AskUserQuestion，通过后 git commit -m "CP3.2: 已完成验收" | |
| 3.3 大 shape 精简 | CP3.2 问卷结果 | developer | 精简后的 CSV + ST 通过 | 仅当用户选择「精简为 1 条」时执行 | |
| **阶段4：上库** | | | | | |
| 4.1 编写文档 | 全部代码和设计文档 | writer | README.md | — | |
| 4.2 代码检视 | git diff + OAT checklist + 全部变更文件 + 文档 | reviewer | 4.2-代码检视报告.md | 变更范围 + OAT 合规复核 + 规范 + 冗余清理 + 日志规范 | |
| 4.3 开发总结 | 全部交付物 | writer | CP4.3.json、4.3-Issue.md、4.3-上库PR模板.md、更新 LOG.md | 整理为问卷 + 提 Issue（内容来自需求文档）+ 生成上库 PR 描述 + 更新开发日志 | |
| ⛔ CP4.3 | CP4.3.json | 用户 | 上库审批 | AskUserQuestion，通过后 squash commit -m "Feat: 新增面向archXX的aclblasXxx接口" | |

**图例**：⛔ 必需确认  ⚪ 仅不通过时直接打回，3次仍失败后询问

## 外部参考仓库

Agent 在架构设计、代码开发和性能优化阶段可按需加载以下外部仓库作为参考：

| 仓库 | 本地路径 | 技能 | 用途 |
|------|---------|------|------|
| [cannbot-skills](https://gitcode.com/cann/cannbot-skills.git) | `.opencode/ref-repos/cannbot-skills/` | 通过 `cannbot_references.json` 映射 | Ascend C 通用技能（API 最佳实践、代码检视、精度调试、性能优化等 16 个 skill） |
| [cann-samples](https://gitcode.com/cann/cann-samples.git) | `.agent/cann-samples/` | `op-samples-reference` | 高性能算子样例、端到端调优实践、SIMT 编程模型参考 |
| [asc-devkit](https://gitcode.com/cann/asc-devkit.git) | `.agent/asc-devkit/` | `asc-devkit-reference` | Ascend C 官方 API 文档（1022+）、示例代码（587+）、实现参考、Tiling 配置 |

初始化时通过 `init.sh` 自动克隆，也可通过 `--cannbot`、`--samples`、`--asc` 参数指定本地路径创建软链接。
