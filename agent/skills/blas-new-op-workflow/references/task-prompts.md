# Task 调用参数

## 通用约束

- **日志摘要不入文档**：每个 Subagent 在回复末尾输出的【日志摘要】段落仅供主 Agent 写入 LOG.md，**不得**写入任何交付文档（.md/json/cpp/h 等）

## 任务恢复映射表

| 中断步骤 | Subagent | 恢复说明 |
|---------|----------|---------|
| 1.1 资料准备 | writer (scene: material-prep) | 读取 LOG.md 继续 |
| 1.2 需求分析 | architect (scene: requirement-analysis) | 读取 LOG.md 继续 |
| 1.3.A 开发方案设计 | architect (scene: design) | 读取 LOG.md 继续 |
| 1.3.B 测试方案设计 | tester (scene: test-design) | 读取 LOG.md 继续 |
| 1.4.A 开发方案评审 | architect (scene: design-review) | 读取 LOG.md 继续 |
| 1.4.B 测试方案评审 | tester (scene: test-design-review) | 读取 LOG.md 继续 |
| 2.0.1 环境准备 | developer | 读取 LOG.md 继续 |
| 2.1.1.A 算子开发 | developer | 读取 LOG.md 继续 |
| 2.1.1.B 测试开发 | tester (scene: test-development) | 读取 LOG.md 继续 |
| 2.1.2 汇合联调 | developer | 读取 LOG.md 继续 |
| 2.1.3 测试验收 | tester (scene: test-execution) | 读取 LOG.md 继续 |
| 2.2.1.A 算子开发 | developer | 读取 LOG.md 继续 |
| 2.2.1.B 测试开发 | tester (scene: test-development) | 读取 LOG.md 继续 |
| 2.2.2 汇合联调 | developer | 读取 LOG.md 继续 |
| 2.2.3 测试验收 | tester (scene: test-execution) | 读取 LOG.md 继续 |
| 3.1 代码检视 | reviewer | 读取 LOG.md 继续 |
| 3.2 性能验收 | developer | 读取 LOG.md 继续 |
| 4.1 编写文档 | writer (scene: write-readme) | 读取 LOG.md 继续 |
| 4.2 代码检视 | reviewer | 读取 LOG.md 继续 |
| 4.3 开发总结 | writer (scene: questionnaire) | 读取 LOG.md 继续 |

## 各阶段 Subagent 调用参数

### 2.0.1 环境准备

```yaml
subagent: developer
输入:
  - 算子名（如 aclSswap）
  - 2.0.1-开发环境.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/2.0.1-开发环境.md)
输出:
  - .claude/dev-docs/{operator_name}/2.0.1-开发环境.md (按模板填写，仅填环境检查项)
  - git 分支，格式为算子名（如 aclSswap）
验收标准:
  - 开发环境检查通过
  - git 分支已创建
  - 未读取任何算子代码/目录/接口信息
  - 若环境有问题（缺少依赖、NPU不可用等），通过 AskUserQuestion 询问用户如何解决
```

### 1.1 资料准备

```yaml
subagent: writer
scene: material-prep
输入:
  - 用户需求描述
  - 用户提供的文档链接列表
  - LOG.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/LOG.md)
  - 参考资料清单模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.1-参考资料清单.md)
  - CP1.1.json 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/CP1.1.json)
输出:
  - .claude/dev-docs/{operator_name}/LOG.md (按模板初始化)
  - .claude/dev-docs/{operator_name}/1.1-参考资料清单.md (按模板填写)
  - .claude/dev-docs/{operator_name}/references/{资料文件名} (下载的网页内容)
  - .claude/dev-docs/{operator_name}/CP1.1.json (按模板填写)
验收标准:
  - 工作区目录 .claude/dev-docs/{operator_name}/ 已创建
  - LOG.md 已初始化
  - 参考资料以条目形式列出（名称 / 位置 / 可参考内容），不展开细节，不做推断
  - 用户提供的链接已下载到 .claude/dev-docs/{operator_name}/references/
  - 若下载失败，已通过问卷要求用户修正链接或同意跳过
  - 接口速查表完整（仅接口名+签名）
  - 禁止搜索仓内目录/代码，禁止使用 grep/find/Glob 查找仓内文件
  - CP1.1.json 中 {aclblasXxx} 已替换，四个问题保留原样不修改结构
  - 目标芯片选项中对应用户指定的芯片添加 "default": true
  - 精度标准选项中根据算子类型标注"（推荐）"：非计算类→位精确，浮点计算类→对应标杆
  - 编程模型选项仅当目标芯片为 arch35 时包含，其他芯片删除此题
```

### 1.2 需求分析

```yaml
subagent: architect
scene: requirement-analysis
输入:
  - CP1.1 对齐结论 (dtype/目标芯片/编程模型/精度标准已在 CP1.1 确认)
  - 1.1-参考资料清单.md
  - 1.2-需求分析.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.2-需求分析.md)
  - CP1.2.json 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/CP1.2.json)
输出:
  - .claude/dev-docs/{operator_name}/1.2-需求分析.md (按模板填写)
  - .claude/dev-docs/{operator_name}/CP1.2.json (按模板填写)
验收标准:
  - 精度标准已对齐
  - 参数约束已记录
  - 可行性评估完整
  - 接口签名/dtype 不再重复确认（已在 CP1.1 对齐）
  - CP1.2.json 中 {aclblasXxx} 和 {operator_name} 已替换，不修改 question/options 结构
```

### 1.3.A 开发方案设计

```yaml
subagent: architect
scene: design
输入:
  - 1.2-需求分析.md
  - 1.3.A-开发方案设计.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.3.A-开发方案设计.md)
输出:
  - .claude/dev-docs/{operator_name}/1.3.A-开发方案设计.md (按模板填写)
验收标准:
  - Tiling 策略完整
  - Kernel 设计明确
  - Host 设计明确
  - API 验证记录完整
  - 参考算子已列出
```

### 1.3.B 测试方案设计

```yaml
subagent: tester
scene: test-design
输入:
  - 1.2-需求分析.md
  - 1.3.B-测试方案设计.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.3.B-测试方案设计.md)
输出:
  - .claude/dev-docs/{operator_name}/1.3.B-测试方案设计.md (按模板填写)
验收标准:
  - L0/L1 用例表完整
  - 精度标准明确
  - 迭代规划清晰
```

### 1.4.A 开发方案评审

```yaml
subagent: architect
scene: design-review
输入:
  - 1.3.A-开发方案设计.md
  - 1.4.A-开发方案评审.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.4.A-开发方案评审.md)
输出:
  - .claude/dev-docs/{operator_name}/1.4.A-开发方案评审.md (按模板填写)
验收标准:
  - 所有评审维度已覆盖
  - 状态字段明确
  - 问题清单完整
```

### 1.4.B 测试方案评审

```yaml
subagent: tester
scene: test-design-review
输入:
  - 1.3.B-测试方案设计.md
  - 1.2-需求分析.md
  - 1.4.B-测试方案评审.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.4.B-测试方案评审.md)
输出:
  - .claude/dev-docs/{operator_name}/1.4.B-测试方案评审.md (按模板填写)
验收标准:
  - 所有评审维度已覆盖
  - 状态字段明确
  - 问题清单完整
```

### 2.1.1.A / 2.2.1.A 算子开发

```yaml
subagent: developer
输入:
  - 1.3.A-开发方案设计.md
输出:
  - blas/{operator_name}/{routine}/archXX/{routine}_host.cpp
  - blas/{operator_name}/{routine}/archXX/{routine}_kernel.cpp
  - blas/{operator_name}/{routine}/archXX/{routine}_tiling_data.h
验收标准:
  - 编译通过
  - 编码规范符合 blas-ascendc-coding-rules
```

### 2.1.1.B / 2.2.1.B 测试开发

```yaml
subagent: tester
scene: test-development
输入:
  - 1.3.B-测试方案设计.md (按迭代指定 L0 或 L0+L1 范围)
  - 加载 blas-ST-develop 技能获取 GTest+CSV 开发规范
输出:
  - test/{operator_name}/{routine}_testcases.csv
  - test/{operator_name}/{routine}_golden.h (CPU 参考实现，不区分架构)
  - test/{operator_name}/arch35/{routine}_test.cpp (架构相关，放 arch35 子目录)
  - test/{operator_name}/CMakeLists.txt
验收标准:
  - CSV 用例覆盖测试设计文档中的所有场景
  - GTest+CSV 参数化模式，自定义 main，ACL 只初始化一次
  - golden.h CPU 参考实现正确
  - CMake 使用 ops_blas_add_gtest_tests
  - 编译通过
```

### 2.1.2 / 2.2.2 汇合联调

```yaml
subagent: developer
输入:
  - 完整算子代码
  - ST 测试用例代码
  - 汇合联调报告模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/2.x.2-汇合联调报告.md)
输出:
  - .claude/dev-docs/{operator_name}/2.1.2-汇合联调报告.md（迭代一）/ 2.2.2-汇合联调报告.md（迭代二）(按模板填写)
验收标准:
  - 编译通过
  - ST 通过率 100%
  - 状态字段 = ✅通过
```

### 2.1.3 / 2.2.3 测试验收

```yaml
subagent: tester
scene: test-execution
输入:
  - 2.1.2-汇合联调报告.md（迭代一）/ 2.2.2-汇合联调报告.md（迭代二）
  - ST 测试工程路径
输出:
  - .claude/dev-docs/{operator_name}/2.1.3-测试验收报告.md（迭代一）/ 2.2.3-测试验收报告.md（迭代二）
验收标准:
  - L0 用例通过率 100%（迭代一）/ L0+L1 全量通过率 100%（迭代二）
  - 状态字段明确
  - 失败用例已记录
```

### 3.1 代码检视

```yaml
subagent: reviewer
输入:
  - 全部算子代码文件路径
输出:
  - .claude/dev-docs/{operator_name}/3.1-代码检视报告.md
验收标准:
  - 代码规范检查完成
  - 风险点已记录
  - 状态字段明确
```

### 3.2 性能验收

```yaml
subagent: developer
输入:
  - 1.2-需求分析.md
  - 1.3.A-开发方案设计.md
  - 3.2-性能报告.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/3.2-性能报告.md)
输出:
  - .claude/dev-docs/{operator_name}/3.2-性能报告.md (按模板填写)
验收标准:
  - 性能指标已采集
  - 瓶颈分析完整
```

### 4.1 编写文档

```yaml
subagent: writer
scene: write-readme
输入:
  - 全部算子代码
  - 全部设计文档
  - README.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/README.md)
输出:
  - blas/{operator_name}/{routine}/README.md
验收标准:
  - 接口说明完整
  - 调用示例正确
  - 编译运行步骤清晰
```

### 4.2 代码检视

```yaml
subagent: reviewer
输入:
  - 全部代码文件路径
  - 全部文档文件路径
输出:
  - .claude/dev-docs/{operator_name}/4.2-代码检视报告.md
验收标准:
  - 规范检查完成
  - 一致性检查完成
  - 风险点已记录
  - 状态字段明确
  - 冗余代码检查：未使用的 #include、未调用的函数/宏、死代码
  - 中间文件清理：删除 .claude/dev-docs/{op}/ 下除 LOG.md 外的所有文件，以及开发过程中产生的临时文件（如 profiler dump、临时脚本等）
  - 交付件清单核对：最终合入的文件集合是最小集
```

### 4.3 开发总结

```yaml
subagent: writer
scene: questionnaire
输入:
  - 全部交付物文件路径
输出:
  - 更新 LOG.md
验收标准:
  - 交付物清单完整
  - 各阶段记录完整
  - 问题记录完整
```

## 日志摘要规范

每个 Subagent 任务完成后，必须在输出末尾追加【日志摘要】段落：

```markdown
---
## 日志摘要（供写入 LOG.md）
- **状态**: ✅完成 / ❌失败
- **关键结论**: 1 行摘要
- **新增文件**: 相对路径列表
- **问题**:
  - 简单问题（1 行可描述）：直接写解决方案
  - 复杂问题：必须已创建 issue 文件，此处只放链接
```

Subagent 不直接修改 LOG.md，由调用方汇总后更新。

## 问题处理

- 简单问题（1 次解决）：日志摘要直接记录
- 复杂问题（多次尝试/需跟进）：创建 `issues/issue_{YYYYMMDD}_{关键词}.md`（模板见 [ISSUE_TEMPLATE.md](../assets/ISSUE_TEMPLATE.md)），LOG.md 只放链接
- 同一任务最多重试 3 次，超过则创建 issue 文件并汇报用户
