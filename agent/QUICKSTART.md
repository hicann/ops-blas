# Blas Agent 快速使用

**Step 1**：初始化

```bash
# Claude Code
bash agent/init.sh claude

# OpenCode
bash agent/init.sh opencode
```

**Step 2**：启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

**Step 3**：描述需求

> 帮我开发一个 ascend950 上的 gemv 算子，支持 FP32 数据类型

---

**特殊情况**

使用本地 cannbot-skills：

```bash
bash agent/init.sh claude --cannbot /path/to/cannbot-skills
```

重新初始化（清空配置目录后重建）：

```bash
bash agent/init.sh claude --clean
```
