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
