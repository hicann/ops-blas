# Task 调用参数

## 通用约束

- **日志摘要不入文档**：每个 Subagent 在回复末尾输出的【日志摘要】段落仅供主 Agent 写入 LOG.md，**不得**写入任何交付文档（.md/json/cpp/h 等）

## 任务恢复映射表

| 中断步骤 | Subagent | 恢复说明 |
|---------|----------|---------|
| 1.1.A 资料准备 | writer (scene: material-prep) | 读取 LOG.md 继续 |
| 1.1.S 总结 | writer (scene: questionnaire) | 读取 LOG.md 继续 |
| 1.1.B 环境准备 | developer | 读取 LOG.md 继续 |
| 1.1.S2 总结 | writer (scene: questionnaire) | 读取 LOG.md 继续 |
| 1.2 需求分析 | architect (scene: requirement-analysis) | 读取 LOG.md 继续 |
| 1.3.A 开发方案设计 | architect (scene: design) | 读取 LOG.md 继续 |
| 1.3.B 测试方案设计 | tester (scene: test-design) | 读取 LOG.md 继续 |
| 1.4.A 开发方案评审 | architect (scene: design-review) | 读取 LOG.md 继续 |
| 1.4.B 测试方案评审 | tester (scene: test-design-review) | 读取 LOG.md 继续 |
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
| 3.3 大 shape 精简 | developer | 读取 LOG.md 继续 |
| 4.1 编写文档 | writer (scene: write-readme) | 读取 LOG.md 继续 |
| 4.2 代码检视 | reviewer | 读取 LOG.md 继续 |
| 4.3 开发总结 | writer (scene: questionnaire) | 读取 LOG.md 继续 |

## 各阶段 Subagent 调用参数

### 1.1.A 资料准备

```yaml
subagent: writer
scene: material-prep
输入:
  - 用户需求描述
  - 用户提供的文档链接列表
  - LOG.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/LOG.md)
  - 参考资料清单模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.1-参考资料清单.md)
输出:
  - .agent/dev-docs/{operator_name}/LOG.md (按模板初始化，operator_name 从用户需求推断临时名称)
  - .agent/dev-docs/{operator_name}/1.1-参考资料清单.md (按模板填写)
  - .agent/dev-docs/{operator_name}/references/{资料文件名} (下载的网页内容)
验收标准:
  - 工作区目录 .agent/dev-docs/{operator_name}/ 已创建（临时名称，CP1.1.A 确认后可能调整）
  - LOG.md 已初始化
  - 参考资料以条目形式列出（名称 / 位置 / 可参考内容），不展开细节，不做推断
  - 用户提供的链接已下载到 .agent/dev-docs/{operator_name}/references/
  - 若下载失败，已通过问卷要求用户修正链接或同意跳过
  - 禁止搜索仓内目录/代码，禁止使用 grep/find/Glob 查找仓内文件
```

### 1.1.S 总结（CP1.1.A.json 生成）

```yaml
subagent: writer
scene: questionnaire
输入:
  - 1.1-参考资料清单.md
  - 用户需求描述
  - CP1.1.A.json 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/CP1.1.A.json)
输出:
  - .agent/dev-docs/{operator_name}/CP1.1.A.json (按模板填写)
验收标准:
  - CP1.1.A.json 中 {aclblasXxx} 已替换为从用户需求推断的 API 名（如 aclblasSgemv）
  - CP1.1.A.json 中 {operator_name} 已替换为从用户需求推断的目录/文件名（snake_case，如 sgeqrf_batched、sgetrf_batched）
  - 算子名问题的 question 文本中 {aclblasXxx} 和 {operator_name} 以及第一个 option 的 label/description 都替换为推断的具体名称
  - 目标芯片选项中对应用户指定的芯片添加 "default": true
  - 不修改 question/options 结构
```

### 1.1.B 环境准备

```yaml
subagent: developer
输入:
  - CP1.1.A 确认的 API 名（如 aclblasSswap）和目录名（如 sswap）
  - 2.0.1-开发环境.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/2.0.1-开发环境.md)
  - LOG.md 文件路径 (.agent/dev-docs/{operator_name}/LOG.md)
输出:
  - .agent/dev-docs/{operator_name}/2.0.1-开发环境.md (按模板填写，仅填环境检查项)
  - git 分支，格式为 {aclblasXxx}（使用 API 名，如 aclblasSswap）
验收标准:
  - 开发环境检查通过
  - git 分支已创建
  - 未读取任何算子代码/目录/接口信息
  - 若环境有问题（缺少依赖、NPU不可用等），通过 AskUserQuestion 询问用户如何解决
```

### 1.1.S2 总结（CP1.1.B.json 生成）

```yaml
subagent: writer
scene: questionnaire
输入:
  - CP1.1.A 对齐结论 (dtype/目标芯片已确认)
  - 1.1-参考资料清单.md
  - CP1.1.B.json 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/CP1.1.B.json)
输出:
  - .agent/dev-docs/{operator_name}/CP1.1.B.json (按模板填写)
验收标准:
  - CP1.1.B.json 中 {aclblasXxx} 和 {operator_name} 已替换
  - 精度标准选项中根据算子类型标注"（推荐）"：非计算类→位精确，浮点计算类→对应标杆
  - 编程模型选项仅当 CP1.1.A 确认的目标芯片为 arch35 时包含，其他芯片删除此题
  - 不修改 question/options 结构
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
  - .agent/dev-docs/{operator_name}/1.2-需求分析.md (按模板填写)
  - .agent/dev-docs/{operator_name}/CP1.2.json (按模板填写)
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
  - 加载 op-samples-reference 技能，参考 cann-samples 仓库中的同类算子架构设计和优化策略选型
  - 加载 asc-devkit-reference 技能，参考 asc-devkit 仓库中的 API 文档、示例代码和 Tiling 配置
输出:
  - .agent/dev-docs/{operator_name}/1.3.A-开发方案设计.md (按模板填写)
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
  - .agent/dev-docs/{operator_name}/1.3.B-测试方案设计.md (按模板填写)
验收标准:
   - L0/L1 用例表完整
   - 精度标准明确
   - 迭代规划清晰
    - 大 shape 用例覆盖：矩阵类 m/n/k >= 1024（至少含 2048 或 4096），向量类 n >= 10000（至少含 100000），不因硬件限制缩减 shape
```

### 1.4.A 开发方案评审

```yaml
subagent: architect
scene: design-review
输入:
  - 1.3.A-开发方案设计.md
  - 1.4.A-开发方案评审.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/1.4.A-开发方案评审.md)
输出:
  - .agent/dev-docs/{operator_name}/1.4.A-开发方案评审.md (按模板填写)
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
  - .agent/dev-docs/{operator_name}/1.4.B-测试方案评审.md (按模板填写)
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
  - 加载 blas-log 技能获取日志集成规范
  - 加载 blas-op-templates 技能获取对应编程模型的代码模板
  - 加载 op-samples-reference 技能，参考 cann-samples 仓库中的同类算子实现和编程模式
  - 加载 asc-devkit-reference 技能，参考 asc-devkit 仓库中的示例代码和 API 文档
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - blas/{family}/{operator_name}/archXX/{operator_name}_host.cpp
  - blas/{family}/{operator_name}/archXX/{operator_name}_kernel.cpp
  - blas/{family}/{operator_name}/archXX/{operator_name}_tiling_data.h
注意:
  - {operator_name} 使用 snake_case 格式（如 sgeqrf_batched、sgetrf_batched），不是 API 名（aclblasSgeqrfBatched）
  - {family} 也使用 snake_case 格式（如 geqrf_batched、getrf_batched）
验收标准:
   - 编译通过
   - 编码规范符合 blas-ascendc-coding-rules（含 R5-R9 OAT 量化规则）
   - OAT 自检通过：所有函数圈复杂度 ≤ 20、函数深度 ≤ 5、NBNC ≤ 50、无除零风险、无 extern 引用告警、源码文件包含标准许可证头
   - OAT checklist 已附在交付报告中（列出各函数的圈复杂度/深度/行数/除零校验/extern 引用检查，标注是否达标）
   - host.cpp 中已集成 dlog 日志：#include "log/log.h"，关键路径使用 OP_LOGE/I/D
   - 参数校验失败、ACL Runtime 调用失败使用 OP_LOGE 输出错误信息
   - Tiling 数据和 Kernel 启动参数使用 OP_LOGD/OP_LOGI 输出
   - 代码以 blas-op-templates 模板为唯一骨架，按设计文档填充业务逻辑
   - 代码结构合规：文件命名、目录布局、类名、函数签名必须与 blas-op-templates 模板一致，禁止从仓内已有算子复制代码骨架（仓内算子可能不符合当前规范）
   - 参考仓内算子时仅提取算法思路（Tiling 切分、数据搬运策略、计算逻辑），禁止复制其文件命名、代码结构或目录布局
   - BLAS 标准对齐：存储顺序为列主序（Column-Major），lda/ldb/ldc 等参数语义与 BLAS 标准一致，禁止参考仓内存储顺序错误的已有算子
```

### 2.1.1.B / 2.2.1.B 测试开发

```yaml
subagent: tester
scene: test-development
输入:
  - 1.3.B-测试方案设计.md (按迭代指定 L0 或 L0+L1 范围)
  - 加载 blas-ST-develop 技能获取 GTest+CSV 开发规范
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - test/{family}/{operator_name}/{operator_name}_param.h (参数结构体，继承 BlasTestParamBase)
  - test/{family}/{operator_name}/{operator_name}_golden.h (CPU golden，签名与 BLAS API 一致)
  - test/{family}/{operator_name}/arch35/{operator_name}_npu_wrapper.h (NPU wrapper，封装 ACL 操作)
  - test/{family}/{operator_name}/arch35/{operator_name}_test.cpp (GTest 入口，5 步流程)
  - test/{family}/{operator_name}/arch35/{operator_name}_test.csv (CSV 用例表，列名=API 参数名)
  - test/{family}/{operator_name}/CMakeLists.txt
注意:
  - {operator_name} 使用 snake_case 格式（如 sgeqrf_batched、sgetrf_batched），与 blas 侧目录名保持一致
  - {family} 也使用 snake_case 格式（如 geqrf_batched、getrf_batched），与 blas 侧 family 目录保持一致
验收标准:
   - CSV 用例覆盖测试设计文档中的所有场景
   - GTest+CSV 参数化模式，BlasTest<Param> fixture，共享 test_main.cpp
    - golden.h / npu_wrapper.h 实现正确
     - golden.h 直接调用 CBLAS/LAPACK 函数（通过 cblas_compat.h），保留参数校验，使用列主序
   - 填充函数只使用 test/frame/fill.h 中已有的公共函数，禁止在测试文件中定义临时填充函数；若需新增，必须补充到 fill.h
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
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - .agent/dev-docs/{operator_name}/2.1.2-汇合联调报告.md（迭代一）/ 2.2.2-汇合联调报告.md（迭代二）(按模板填写)
验收标准:
   - 编译通过
   - ST 通过率 100%
   - 状态字段 = ✅通过
   - 测试代码未被修改：联调过程中禁止删除或修改测试用例（CSV 行、TEST_P）、golden.h 计算逻辑、npu_wrapper.h 封装逻辑
```

### 2.1.3 / 2.2.3 测试验收

```yaml
subagent: tester
scene: test-execution
输入:
  - 2.1.2-汇合联调报告.md（迭代一）/ 2.2.2-汇合联调报告.md（迭代二）
  - ST 测试工程路径
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - .agent/dev-docs/{operator_name}/2.1.3-测试验收报告.md（迭代一）/ 2.2.3-测试验收报告.md（迭代二）
验收标准:
   - L0 用例通过率 100%（迭代一）/ L0+L1 全量通过率 100%（迭代二），不允许有任何失败
   - 状态字段明确
   - 失败用例已记录（若有）
   - 测试代码完整性验证（强制）：
      - 检查 CSV 文件行数与测试设计文档中的用例数一致
      - 检查 param.h/golden.h/npu_wrapper.h/test.cpp/test.csv 的 git diff，确认无未授权修改
      - 若发现测试代码被篡改，立即标记验收失败，打回开发侧重新联调
```

### 3.1 代码检视

```yaml
subagent: reviewer
输入:
  - git diff --stat 输出（主 Agent 执行 `git diff --stat cann/master...HEAD` 后的完整结果）
  - git diff --name-only 列出的所有变更文件路径（不仅是算子代码，包括共享文件）
  - developer 交付报告中的 OAT checklist（各函数圈复杂度/深度/行数/除零/extern 自检结果）
  - 3.1-代码检视报告.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/3.1-代码检视报告.md)
输出:
  - .agent/dev-docs/{operator_name}/3.1-代码检视报告.md
验收标准:
   - 变更范围检查（强制，第一步）：reviewer 必须自行执行 `git diff --stat cann/master...HEAD` 进行独立验证（不依赖主 Agent 传递的 diff 输出），逐文件确认：
     （1）共享文件（aclblas_kernel_do.h、cann_ops_blas.h、fill.h、csv_loader.h）的修改仅包含本算子的新增内容，不得包含对已有代码的格式化、重排、删除；
     （2）不存在与本算子无关的文件变更。
     发现无关变更必须标记为 HIGH 问题并要求还原。
   - OAT 合规复核：对照 developer 的 OAT checklist，抽查关键函数的圈复杂度/深度/行数是否达标，发现遗漏或超标标记为 HIGH 问题
   - 代码规范检查完成
   - 风险点已记录
   - 状态字段明确
   - 日志规范检查：host.cpp 中日志级别使用正确、消息格式符合 blas-log 规范
   - 冗余代码检查（HIGH 置信度）：未使用的 #include、未调用的函数/宏、未使用的变量/参数、死代码、重复定义，发现即要求删除
```

### 3.2 性能验收

```yaml
subagent: developer
输入:
  - 1.2-需求分析.md
  - 1.3.A-开发方案设计.md
  - 3.2-性能报告.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/3.2-性能报告.md)
  - 加载 op-samples-reference 技能，参考 cann-samples 仓库中的性能调优实践和瓶颈分析方法
  - 加载 asc-devkit-reference 技能，参考 asc-devkit 仓库中的性能优化实践和瓶颈分析方法
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - .agent/dev-docs/{operator_name}/3.2-性能报告.md (按模板填写)
验收标准:
  - 性能指标已采集
  - 瓶颈分析完整
  - 性能测试的中间文件/结果文件统一存放在 test/{family}/{operator_name}/perf/ 目录下
  - 性能分析结束后及时删除 test/{family}/{operator_name}/perf/ 下的所有中间文件和结果文件
```

### 3.3 大 shape 精简

```yaml
subagent: developer
输入:
  - CP3.2.ret.json（用户选择「精简为 1 条」时触发）
  - 当前测试 CSV 文件路径
  - 加载 blas-build-commands 技能获取编译和验证命令
输出:
  - 精简后的 CSV 文件（仅保留 1 条代表性大 shape 用例）
验收标准:
   - CSV 中大 shape 用例仅保留 1 条
   - 保留的用例具有代表性（覆盖最大或典型大 shape）
   - ST 编译通过且通过率 100%
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
  - blas/{family}/{operator_name}/README.md
验收标准:
  - 接口说明完整
  - 调用示例正确
  - 编译运行步骤清晰
```

### 4.2 代码检视

```yaml
subagent: reviewer
输入:
  - git diff --stat 输出（主 Agent 执行 `git diff --stat cann/master...HEAD` 后的完整结果）
  - git diff --name-only 列出的所有变更文件路径（不仅是算子代码，包括共享文件和文档）
  - developer 交付报告中的 OAT checklist（各函数圈复杂度/深度/行数/除零/extern 自检结果）
  - 全部文档文件路径
  - 4.2-代码检视报告.md 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/4.2-代码检视报告.md)
输出:
  - .agent/dev-docs/{operator_name}/4.2-代码检视报告.md
验收标准:
   - 变更范围检查（强制，第一步）：reviewer 必须自行执行 `git diff --stat cann/master...HEAD` 进行独立验证（不依赖主 Agent 传递的 diff 输出），逐文件确认：
     （1）共享文件（aclblas_kernel_do.h、cann_ops_blas.h、fill.h、csv_loader.h）的修改仅包含本算子的新增内容，不得包含对已有代码的格式化、重排、删除；
     （2）不存在与本算子无关的文件变更。
     发现无关变更必须标记为 HIGH 问题并要求还原。
   - OAT 合规复核：对照 developer 的 OAT checklist，抽查关键函数的圈复杂度/深度/行数是否达标，发现遗漏或超标标记为 HIGH 问题
   - 规范检查完成
   - 一致性检查完成
   - 风险点已记录
   - 状态字段明确
   - 冗余代码检查（HIGH 置信度，零容忍）：未使用的 #include、未调用的函数/宏、未使用的变量/参数、死代码、重复定义，发现即要求删除
   - 交付件清单核对：最终合入的文件集合是最小集，不含任何未使用的文件
   - 日志规范检查：无残留 printf/LOG_PRINT，全部使用 OP_LOGE/I/D/W
```

### 4.3 开发总结

```yaml
subagent: writer
scene: questionnaire
输入:
  - 全部交付物文件路径
  - CP4.3.json 模板文件路径 (模板路径: agent/skills/blas-new-op-workflow/assets/CP4.3.json)
  - PR 模板文件路径 (模板路径: agent/skills/gitcode-pr-issue-guide/assets/PULL_REQUEST_TEMPLATE.zh-CN.md)
  - LOG.md（用于提取开发过程摘要）
  - 1.2-需求分析.md（用于提取算子功能描述，同时作为 Issue 文本来源）
  - 1.3.A-开发方案设计.md（用于提取 Tiling/Kernel 设计思路，作为 Issue Design 字段来源）
  - 2.1.3-测试验收报告.md / 2.2.3-测试验收报告.md（用于提取测试结论）
  - 3.2-性能报告.md（用于提取性能结论）
输出:
  - .agent/dev-docs/{operator_name}/CP4.3.json (按模板填写)
  - .agent/dev-docs/{operator_name}/4.3-Issue.md (Issue 文本，内容来自 1.2-需求分析.md)
  - .agent/dev-docs/{operator_name}/4.3-上库PR模板.md (按 PR 模板填写，见下方填写规范)
  - 更新 LOG.md
验收标准:
  - 交付物清单完整
  - 各阶段记录完整
  - 问题记录完整
  - CP4.3.json 中 {aclblasXxx}、{archXX} 和 {operator_name} 已替换，不修改 question/options 结构
  - 4.3-Issue.md 内容来自 1.2-需求分析.md，包含算子功能描述、参数约束、精度标准等需求信息
  - 4.3-上库PR模板.md 各字段已填写（见下方填写规范）
```

**4.3-Issue.md 填写规范：**

Issue 文本的内容**必须来自 1.2-需求分析.md**（算子最开始设计的需求文档），而非自行编写。使用 `gitcode-pr-issue-guide` 技能的**需求建议模板**（`feature-request.yml`），按以下规则填充：

| 字段 | 填充来源 | 说明 |
|------|---------|------|
| Issue 标题 | 固定格式 | `Feat: 新增面向{目标芯片}的aclblas{Xxx}接口` |
| Background（背景信息） | 1.2-需求分析.md | 算子功能描述、参数约束、精度标准、目标芯片及 CANN 版本 |
| Origin（信息来源） | 固定值 | `cann 开发者` |
| Benefit / Necessity | 1.2-需求分析.md | 应用场景、需求价值 |
| Design（设计方案） | 1.3.A-开发方案设计.md | Tiling/Kernel 设计思路概述 |

**提交流程**：4.3-Issue.md 生成后，通过 `gitcode-pr-issue-guide` 技能提交到 GitCode，获取 Issue URL，然后将该 URL 填入 4.3-上库PR模板.md 的"关联的Issue"字段。

**4.3-上库PR模板.md 填写规范：**

读取 `agent/skills/gitcode-pr-issue-guide/assets/PULL_REQUEST_TEMPLATE.zh-CN.md` 模板，按以下规则填充各字段：

| 字段 | 填充来源 | 说明 |
|------|---------|------|
| 描述 | 1.2-需求分析.md + 交付物清单 | 概述算子功能、目标芯片/dtype、实现方法（Tiling策略、Kernel结构） |
| 关联的Issue | 4.3-Issue.md 提交后的 Issue 链接 | 通过 `gitcode-pr-issue-guide` 技能提交 Issue 后，将返回的 Issue URL 填入此处 |
| 测试 | 2.2.3-测试验收报告.md + 3.2-性能报告.md | 列出 ST 通过率、精度标准、性能数据 |
| 文档更新 | 交付物清单中的文档部分 | 列出新增/修改的文档文件 |
| 类型标签 | 固定选"新特性" | 算子新增属于新特性 |

输出为 md 格式，保留模板的 `##` 标题结构，删除 HTML 注释，直接填充内容。

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
