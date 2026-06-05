# Token 获取方式问卷

**触发条件**：需要调用 GitCode API 时，**且** `~/.git-credentials` 中未找到 gitcode.com 的 token

> 注意：本问卷是兜底方案。正常流程应先自动读取 `~/.git-credentials`，仅在读取失败时才发送本问卷。

```
问题：未从 ~/.git-credentials 中找到 GitCode token，请手动提供：
选项：
- 我来手动提供 token（用户在自定义输入中粘贴）
```

**后续处理**：
- 等待用户在自定义输入中粘贴 token

**安全规则**：获取到 token 后保存为变量 `TOKEN`，**禁止**将 token 打印到输出或写入任何文件。
