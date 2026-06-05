# 修改 init.sh

> 适用于：`agent/init.sh`

## 修改部署脚本

1. 修改 `agent/init.sh`
2. 检查 `blas-new-op-workflow-maintain/SKILL.md` 中的「部署原理（init.sh）」表格是否需要更新
3. 检查 `agent/QUICKSTART.md` 中的用法示例是否需要更新
4. 执行 `references/common.md` 通用检查

## init.sh 步骤表（当前版本）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 创建 `.opencode/` 或 `.claude/` + `.agent/dev-docs/` | 根据 target 参数选择，同时创建临时文档目录 |
| 2 | 软链接 `AGENT.md` | `AGENTS.md -> agent/AGENT.md`（opencode）或 `CLAUDE.md -> agent/AGENT.md`（claude） |
| 3 | 软链接 agents | `.opencode/agents/*.md -> agent/agents/*.md` |
| 4 | 设置 cannbot-skills | clone 或使用本地路径 |
| 5 | 软链接本地 skills | `.opencode/skills/* -> agent/skills/*` |
| 6 | 软链接 cannbot skills | 读取 `cannbot_references.json`，从 cannbot-skills 仓库软链接 |
| 7 | 设置外部参考仓库 | clone cann-samples 和 asc-devkit 到 `.agent/` |

## 关键参数

| 参数 | 说明 |
|------|------|
| `<target>` | `claude` 或 `opencode`，决定运行时目录 |
| `--clean` | 清空 `.claude/`、`.opencode/`、`.agent/` 后重建（或仅清空，不初始化） |
| `--cannbot <path>` | 使用本地 cannbot-skills 路径 |
| `--samples <path>` | 使用本地 cann-samples 路径 |
| `--asc <path>` | 使用本地 asc-devkit 路径 |
