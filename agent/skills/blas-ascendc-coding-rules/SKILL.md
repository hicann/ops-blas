---
name: blas-ascendc-coding-rules
description: ops-blas 仓 AscendC 编码规范，提供常见编码约束的错误示例和正确示例。触发：需要查询编码规范、违反编码约束时报错查纠。
---

# AscendC 编码规范

| 编号 | 规则 | 说明 |
|------|------|------|
| R1 | 禁止逐元素操作 | 禁止 SetValue/GetValue、逐元素 DataCopy → [R1-禁止逐元素操作.md](references/R1-禁止逐元素操作.md) |
| R2 | 动态获取 CoreNum | 禁止硬编码核数，使用 aclrtGetDeviceInfo → [R2-动态获取CoreNum.md](references/R2-动态获取CoreNum.md) |
| R3 | TPipe 禁止作为成员变量 | TPipe 应在 Kernel 入口创建 → [R3-TPipe禁止成员变量.md](references/R3-TPipe禁止成员变量.md) |
| R4 | TilingData 禁止使用数组 | 不可用数组为每个核分配独立地址，应在 TilingData 中记载核间间隔，由 Kernel 根据 blockIdx 自行计算 → [R4-TilingData禁止数组.md](references/R4-TilingData禁止数组.md) |
| R5 | 圈复杂度 ≤ 20 | 函数的 Cyclomatic Complexity 不超过 20，超过需拆分子函数 |
| R6 | 函数深度 ≤ 5 | 最大嵌套层级不超过 5，超过需提取内层循环/分支为独立函数 |
| R7 | 函数行数 ≤ 50 | NBNC（非空非注释行）不超过 50，超过需拆分为多个函数 |
| R8 | 除零防御 | 除法/取模运算的被除数必须校验非零，特别是来自外部输入的变量（如 coreNum） |
| R9 | 许可证头 | 所有源码文件必须包含标准许可证头（CSV 文件除外） |
| R10 | 禁止 extern 引用 | 禁止在 Kernel/Host 代码中使用 `extern "C"` 声明外部函数接口，应通过头文件 include 引入 |
