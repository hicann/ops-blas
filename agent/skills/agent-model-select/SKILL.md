---
name: agent-model-select
description: |
  为工作流 agent 配置合适的模型。触发场景：用户要求配置 agent 模型、选择模型、修改 agent 使用的模型、
  询问哪个模型最好、模型选型建议。关键词：配置模型、选模型、model config、换个模型、模型推荐。
---

# Agent 模型配置技能

## 概述

本技能指导主 Agent 为工作流中的各 Subagent 配置合适的模型。配置写入 `agent/agents/model_config.json`，
通过 `init.sh` 生成项目级 `opencode.json` 生效。

## 获取 Agent 列表

**禁止硬编码 agent 列表**。所有 agent 的名称和描述必须从 `agent/AGENT.md` 的 `agents:` 字段动态读取，
每个 agent 的角色描述从对应的 `agent/agents/<name>.md` 的 frontmatter `description` 字段获取。

## 模型选型原则

根据每个 agent 的 `description` 判断其核心能力需求，按以下原则选型：

| 能力需求 | 选型方向 |
|---------|---------|
| 复杂推理、需求分析、方案设计 | 旗舰推理模型 |
| 代码生成、调试、性能优化 | 旗舰代码模型 |
| 代码审查、规范检查、细致分析 | 中高端推理模型 |
| 测试代码生成、用例设计 | 代码模型（中大型） |
| 文档整理、结构化输出 | 中端模型或 default |

### 模型类型识别

从 `opencode models` 输出中识别模型类型：
- **推理模型**：模型名中包含 `max`、`reasoner`、`thinking`、`r1` 等关键词，或参数量大（如 480b）
- **代码模型**：模型名中包含 `coder`、`code` 等关键词
- **通用模型**：模型名中包含 `plus`、`turbo`、`flash` 等关键词

### 选型策略

根据用户偏好选择不同策略：

- **性能优先**（不考虑成本）：所有 agent 都用旗舰级
  - 推理/审查类角色 → 旗舰推理模型
  - 编码/测试类角色 → 旗舰代码模型
- **平衡**：核心角色用旗舰，辅助角色用 default
  - 架构/开发类角色 → 旗舰
  - 其余角色 → default
- **成本优先**：仅开发类角色用旗舰代码模型，其余 default

## 配置流程

### Step 1：获取可用模型列表和 Agent 列表

1. 运行 `opencode models` 获取可用模型列表
2. 读取 `agent/AGENT.md` 的 `agents:` 字段获取所有 agent 名称
3. 读取每个 `agent/agents/<name>.md` 的 frontmatter 获取 `description`

### Step 2：发送问卷确认用户偏好

使用 `question` 工具发送问卷，询问用户选型偏好：

```
问题：请选择模型选型策略
选项：
- 性能优先：所有 agent 都用最强模型，不考虑成本
- 平衡：核心角色用旗舰模型，其余使用默认
- 成本优先：仅开发类角色用旗舰代码模型，其余使用默认
- 逐个指定：手动为每个 agent 指定模型
```

若用户选择「逐个指定」，则依次询问每个 agent 的模型选择。

### Step 3：根据偏好选择模型

根据用户选择的策略，从可用模型列表中挑选最合适的模型：

1. 识别可用模型中的推理模型和代码模型
2. 按参数量/能力等级排序（通常模型名中的数字越大能力越强）
3. 根据每个 agent 的 description 判断其能力需求，按选型策略分配模型

### Step 4：写入配置

**必须为所有 agent 都写入配置**，不可遗漏任何一个。

写入 `agent/agents/model_config.json`，格式为：

```json
{
  "<agent-name>": {
    "comment": "<agent 的中文角色描述>",
    "model": "provider/model-id"
  }
}
```

**规则**：
- 所有 agent 都必须配置，使用 `"default"` 表示跟随主 Agent 模型
- `model` 字段格式为 `provider/model-id`，必须在可用模型列表中
- `comment` 字段从 agent 的 description 中提取中文角色名
- 主 Agent（build）的模型在 opencode 启动时选择，无需在此配置

### Step 5：使配置生效

自行运行 `bash agent/init.sh <target>` 使配置生效，**禁止**让用户退出当前会话重新运行 init。

init.sh 会自动：
1. 校验模型是否在可用列表中（不可用则回退 default 并 warning）
2. 将非 default 配置写入项目级 `opencode.json`
3. 全部为 default 时删除 `opencode.json`

### Step 6：发送问卷确认各 agent 配置

配置生效后，使用 `question` 工具发送问卷，动态列出所有 agent 的当前配置，询问用户是否需要修改：

```
问题：以下是各 agent 的模型配置，是否需要修改？

当前配置：
- <agent-name>（<角色描述>）: <model>
- ...

选项：
- 确认，无需修改
- 需要修改 <agent-name> 的模型
- ...
```

允许多选。若用户选择了需要修改的 agent，则针对选中的 agent 发送新问卷，让用户从可用模型列表中选择新模型，更新配置后重新执行 Step 5。

### Step 7：确认结果

向用户展示最终配置和生效状态。

## 重要说明

- **opencode.json 不需要重启**：Subagent 模型在每次调用时读取，修改后下次调用即生效
- **主 Agent 模型**：启动时选定，修改需要重启 opencode
- **model_config.json 在 .gitignore 中**：本地配置，不会被 git 追踪
- **重置配置**：删除 `agent/agents/model_config.json` 后运行 init.sh 即可恢复全部 default
