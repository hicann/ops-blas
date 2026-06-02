# Token 获取方式问卷

**触发条件**：需要调用 GitCode API 时

```
问题：如何获取你的 GitCode access token？
选项：
- 从 ~/.git-credentials 文件自动读取（推荐）
- 我来手动提供 token（用户在自定义输入中粘贴）
```

**后续处理**：
- 选择"自动读取"：执行 `cat ~/.git-credentials`，提取 `gitcode.com` 对应的 token（URL 中 `:` 和 `@` 之间的部分）
- 选择"手动提供"：等待用户在自定义输入中粘贴 token

**安全规则**：获取到 token 后保存为变量 `TOKEN`，**禁止**将 token 打印到输出或写入任何文件。
