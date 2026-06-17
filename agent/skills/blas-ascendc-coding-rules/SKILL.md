---
name: blas-ascendc-coding-rules
description: |
  ops-blas 仓 AscendC 编码规范 + MR 安全编码规则速查索引。
  **非开发期常驻上下文**：作为代码提交前的自查工具，按需读取。
  触发时机：① 2.x.2 联调前 ② 2.x.3 验收前 ③ 3.1/4.2 代码检视前 ④ reviewer 检视 MR 代码时。
  详细规则内容位于 references/ 目录，由调用方按需加载相关文档。
---

# AscendC 编码规范速查索引

本规则为**自查清单**，不加载为开发期常驻上下文。仅在以下节点由 agent 按需调用：

| 调用时机 | 适用角色 | 调用方式 |
|---------|---------|---------|
| 2.x.1.A 算子初稿完成后 | developer | 加载本 skill，按 `references/checklist.md` 自查一次 |
| 2.x.2 联调前 | developer | 同上 |
| 2.x.3 测试验收前 | developer | 同上 + 加载 `references/mr-rules-essential.md` 做 MR 合规审查 |
| CP2.x 打点前 | developer | 同上 |
| 3.1 / 4.2 代码检视前 | reviewer | 加载本 skill，对照 `references/mr-rules-essential.md` + `references/mr-rules-general.md` 做检视 |

## references 索引

### AscendC 编码规则（R1-R10）

- 每条规则独立成一个 reference 文件，包含错误/正确示例：

| 编号 | 规则 | reference |
|------|------|-----------|
| R1 | 禁止逐元素操作 | `references/R1-禁止逐元素操作.md` |
| R2 | 动态获取 CoreNum（禁止硬编码） | `references/R2-动态获取CoreNum.md` |
| R3 | TPipe 禁止作为成员变量 | `references/R3-TPipe禁止成员变量.md` |
| R4 | TilingData 禁止使用数组做核间分配 | `references/R4-TilingData禁止数组.md` |
| R5-R10 | 圈复杂度/嵌套深度/函数行数/除零防御/许可证头/extern 引用 | `references/ascendc-r5-r10.md` |

### MR 安全编码规则

- 按**严重等级**分组，方便 reviewer 按优先级检视：

| reference | 包含内容 | 适用场景 |
|-----------|---------|---------|
| `references/mr-rules-essential.md` | 严重/致命级规则（G.PRE.05、G.INC.*、G.FUU.09/10/12/13/15、G.MEM.04、G.STD.*、OAT 等 ~18 条） | MR 提交前必查 |
| `references/mr-rules-general.md` | 一般/建议级规则（G.EXP.*、G.CTL.03、G.AST.03、G.FUU.11/14、CQ.*、CIP.01 等 ~17 条） | 代码检视深度审查 |

### 检查流程与修复指南

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/checklist.md` | 8 步检查流程（文件级 → 头文件 → 函数级 → 表达式 → 安全函数 → 内存 → 标准库 → 冗余告警） | 提交前系统自查 |
| `references/fix-guide.md` | 14 种常见违规的修复方法对照表 | 发现违规后查修复方案 |

## 使用示例

### 场景 1：developer 自查

在 2.x.1.A 完成后，调用本 skill 执行自检：

```
1. 加载 `references/checklist.md`，按 8 步流程逐条检查代码
2. 发现 R6 嵌套深度超标 → 加载 `references/ascendc-r5-r10.md` 查看修复方法
3. 发现 G.FUU.09 使用了 realloc → 加载 `references/mr-rules-essential.md` 确认级别与修复方案
4. 修复完成后再次执行 checklist.md 直至通过
```

### 场景 2：reviewer 检视

在 3.1 / 4.2 检视前，加载：

```
1. `references/mr-rules-essential.md` 对照严重/致命规则做检视（必须零违规）
2. `references/mr-rules-general.md` 做深度审查（建议修复）
3. `references/ascendc-r5-r10.md` 检查代码质量指标（圈复杂度/嵌套深度/行数）
```

---

**注意**：本 skill 的 SKILL.md 仅作为索引与触发说明。Agent 在执行自查/检视时，应直接读取相应的 references 文件获取完整规则，**不要**在开发过程中常驻本 skill 的全部内容。
