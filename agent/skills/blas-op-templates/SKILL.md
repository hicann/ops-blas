# Skill: blas-op-templates

ops-blas 算子代码模板库，为不同编程模型和目标架构提供标准化的代码骨架。Agent 在开发新算子时，应以对应类型的模板为起点，按算子需求填充业务逻辑。

---

## 算子分类体系

ops-blas 仓中的算子按 **编程模型** 和 **目标架构** 两个维度分类：

### 编程模型

| 编程模型 | 关键特征 | 适用场景 | 模板目录 |
|---------|---------|---------|---------|
| **SIMD membase** | `TPipe` / `TQue` / `DataCopyPad` / `SetFlag/WaitFlag<HardEvent::...>` | 数据搬运为主、Vector 计算为辅的算子（Level-1 BLAS 等） | `references/simd-membase/` |
| **SIMD regbase** | `__VEC_SCOPE__` / `RegTensor` / `MicroAPI::DataCopy/Mul/Add/ReduceSum` | 寄存器级 SIMD 算子，在 UB 内使用寄存器张量计算 | `references/simd-regbase/` |
| **SIMT** | `__simt_vf__` / `asc_vf_call` / `threadIdx.x` / `blockDim.x` | 线程级并行算子（Level-2/3 BLAS 等，仅 arch35） | `references/simt/` |

### 目标架构

| 架构 | SOC_VERSION | NPU_ARCH | 说明 |
|------|------------|----------|------|
| arch20 | ascend310p* | dav-1101 | 推理芯片 |
| arch22 | ascend910b* / ascend910_93* | dav-2201 | 训练/推理芯片 |
| **arch35** | **ascend950*** | **dav-3510** | **Atlas A5 系列（当前重点）** |

---

## 模板目录结构

每个编程模型的模板文件按仓库实际目录层级组织，从 `blas/` 一级开始，仅包含算子实现代码：

```
references/<programming-model>/
  blas/{family}/{op}/
    archxx/
      op_tiling_data.h       -- Tiling 数据结构（host/kernel 共享）
      op_kernel.cpp          -- Device 侧 kernel 实现
      op_host.cpp            -- Host 侧 API 入口 + Tiling 计算
```

说明：
- `simd-membase` 模板适用于 arch22 和 arch35，使用 `archxx/` 作为通用目录名
- `simd-regbase` 模板仅适用于 arch35（DAV_3510 RegBase 模式），使用 `arch35/` 目录名
- `simt` 模板仅适用于 arch35（arch22 不支持 SIMT 编程模型）
- `{family}` 和 `{op}` 为占位符，使用时替换为实际的算子族名和算子名（如 `swap/sswap`、`dot/sdot`）

---

## 算子交付件目录结构

一个完整算子在仓库中的文件布局（模板仅覆盖 `blas/` 部分）：

```
blas/<family>/<op_name>/
  README.md
  arch35/
    <op_name>_host.cpp
    <op_name>_kernel.cpp
    <op_name>_tiling_data.h
```

---

## 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 算子目录 | `blas/<family>/<blas_name>/` | `blas/swap/sswap/` |
| Kernel 文件 | `<op>_kernel.cpp` | `sswap_kernel.cpp` |
| Host 文件 | `<op>_host.cpp` | `sswap_host.cpp` |
| Tiling 头文件 | `<op>_tiling_data.h` | `sswap_tiling_data.h` |
| Tiling 结构体 | `<Op>TilingData`（PascalCase） | `SswapTilingData` |
| Kernel 类 | `<Op>AIV` 或 `<Op>Kernel<T>`（SIMD） | `SswapAIV` |
| SIMT 计算函数 | `<Op>SimtCompute` + `__simt_vf__` | `SspmvSimtCompute` |
| Kernel 入口 | `<op>_kernel`（`__global__`） | `sswap_kernel` |
| Kernel 启动器 | `<op>_kernel_do` | `sswap_kernel_do` |
| 公共 API | `aclblas<Op>`（PascalCase） | `aclblasSswap` |
| 测试 fixture | `<Op>Arch35Test` | `SswapArch35Test` |

---

## 使用方法

1. 根据算子的编程模型和目标架构，选择对应的模板目录
2. 将模板文件复制到算子目录，按命名规范重命名
3. 按模板中的 `// TEMPLATE:` 注释指引，替换占位逻辑为实际业务逻辑
4. 模板中的代码已包含标准框架（Init/Process、参数校验、Tiling 计算等），只需填充核心算法部分

---

## 参考资源

| 资源 | 路径 | 说明 |
|------|------|------|
| SIMD membase 模板 | [references/simd-membase/](references/simd-membase/) | 以 sswap 为原型抽取的 SIMD membase 算子全套模板（arch22/arch35 通用） |
| SIMD regbase 模板 | [references/simd-regbase/](references/simd-regbase/) | 以 gemv_batched 为原型抽取的 SIMD regbase 算子全套模板（仅 arch35） |
| SIMT 模板 | [references/simt/](references/simt/) | 以 sspmv/sgemv 为原型抽取的 SIMT 算子全套模板（仅 arch35） |

### 原始参考算子

| 编程模型 | 参考算子 | 路径 |
|---------|---------|------|
| SIMD membase | sswap | `blas/swap/sswap/arch35/` |
| SIMD membase | sdot | `blas/dot/sdot/arch35/` |
| SIMD regbase | gemv_batched | `blas/gemv_batched/arch35/` |
| SIMT | sspmv | `blas/spmv/sspmv/arch35/` |
| SIMT | sgemv | `blas/gemv/sgemv/arch35/` |

Base directory for this skill: file:///mnt/workspace/gitCode/cann/ops-blas/agent/skills/blas-op-templates
