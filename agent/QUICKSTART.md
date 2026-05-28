# Blas Agent 快速使用

**Step 1**：初始化

```bash
bash agent/init.sh claude
```

**Step 2**：启动

```bash
claude
```

**Step 3**：描述需求

> 帮我开发一个 ascend950 上的 gemv 算子，支持 FP32 数据类型

---

**特殊情况**

使用本地 cannbot-skills：

```bash
bash agent/init.sh claude --cannbot /path/to/cannbot-skills
```

重新初始化（清空 .claude/ 后重建）：

```bash
bash agent/init.sh claude --clean
```
