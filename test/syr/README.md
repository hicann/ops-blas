## Syr算子实现

## 概述

本样例展示 `aclblasSsyr` 在 Ascend A5 平台上的基本使用流程。

Syr (Symmetric Rank-1 Update) 实现对称矩阵的秩-1更新操作，数学表达式为：

```
A := alpha * x * x^T + A
```

其中 A 为 n×n 对称矩阵，仅上三角或下三角区域被引用和更新。

## 支持的产品

- Atlas A5 系列产品 (Ascend950)

## 目录结构介绍

```
├── syr
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── syr_test.cpp        // 算子调用样例
```

## 算子描述

- 算子功能：
  Syr算子实现对称秩-1更新操作，将 `alpha * x * x^T` 加到对称矩阵 `A` 的指定三角区域。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Ssyr</td></tr>
  <tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">A</td><td align="center">n x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">scalar</td><td align="center">float32</td><td align="center">-</td></tr>
  <tr><td align="center">uplo</td><td align="center">scalar</td><td align="center">enum</td><td align="center">ACLBLAS_UPPER / ACLBLAS_LOWER</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">A</td><td align="center">n x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">syr_kernel</td></tr>
  </table>

- 调用实现：
  本样例为 Host API 调用示例，使用 `aclblasSsyr` 接口完成算子配置与执行。

- 接口定义：

```cpp
aclblasStatus_t aclblasSsyr(aclblasHandle handle,
                             aclblasFillMode uplo,
                             const int64_t n,
                             const float *alpha,
                             const float *x, const int64_t incx,
                             float *A, const int64_t lda);
```

- 参数说明：

| 参数 | 说明 |
|------|------|
| handle | aclBLAS 库句柄 |
| uplo | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域 |
| n | 矩阵阶数，n >= 0 |
| alpha | 标量乘数指针 |
| x | 输入向量指针，长度至少 `1 + (n-1) * abs(incx)` |
| incx | x 的元素间步长，incx != 0 |
| A | 对称矩阵指针，维度 (lda, n) |
| lda | A 的主维度，lda >= max(1, n) |

- 返回码说明：

| 返回码 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 执行成功 |
| ACLBLAS_STATUS_INVALID_VALUE | 参数无效（n<0, incx=0, lda不足, 空指针等） |
| ACLBLAS_STATUS_ALLOC_FAILED | 内存分配失败 |
| ACLBLAS_STATUS_INTERNAL_ERROR | 内部执行错误 |

- 精度指标：

| 数据类型 | atol | rtol |
|---------|------|------|
| float32 | 0.002 | 0.001 |

## 关键设计

- 多核并行：循环行分布 (Cyclic Row Distribution)，多核负载均衡比约 1.08:1
- 数据搬运：使用标准 Ascend C API (DataCopy/DataCopyPad)，三阶流水线 (TPipe/TQue)
- TILE_SIZE 动态化：按 UB 容量自适应，减少 GM 交互次数
- incx != 1：Host 侧处理非连续 stride，将数据重组为连续 buffer 再传入 Kernel

## 编译运行

- 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

- 样例执行

```bash
bash build.sh --ops=syr --soc=ascend950 --run
```

执行结果如下，说明精度对比成功。

```
========================================
  Total: 21  Passed: 21  Failed: 0
========================================
  RESULT: ALL TESTS PASSED
```

## 注意事项

1. 本算子仅支持 Ascend950 (A5) 平台，旧版 ssyr 算子已从 A5 排除编译
2. 接口对齐 cuBLAS cublasSsyr，alpha 为指针类型 `const float *`
