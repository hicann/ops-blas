## GemvBatched 算子实现

## 概述

BLAS GemvBatched 算子实现。

GemvBatched（批量实数矩阵-向量乘法）实现了对一批矩阵分别进行矩阵-向量乘法的运算，是 BLAS Level 2 核心算子之一。针对 Ascend 950（arch35）架构，支持 S（FP32 入/出）、HSH（FP16 入/出）、HSS（FP16 入/FP32 出）三种精度。

## 支持的产品

| 产品                                 | 是否支持 |
| :----------------------------------- | :------: |
| Ascend 950PR / Ascend 950DT          |    ✓     |

## 目录结构介绍

```
blas/gemv_batched/
├── README.md                           // 说明文档
├── arch35/
│   ├── gemv_batched_host.cpp           // Host 侧实现（参数校验、Tiling 计算、Kernel 调用）
│   ├── gemv_batched_kernel.cpp         // Kernel 侧实现（AIV SIMD 批量运算 + SIMT 转置路径）
│   └── gemv_batched_tiling_data.h      // Tiling 数据结构（Host 和 Kernel 共用）
└── cgemv_batched/
    └── arch22/                         // 复数批量 GEMV（arch22 实现）
```

## 算子描述

- 算子功能：

GemvBatched 对每个 batch 独立完成矩阵-向量乘法。对应的数学表达式为：

```
y[i] = alpha * op(A[i]) * x[i] + beta * y[i]
```

其中 `op(A)` 可以是：
- `A`（不转置，trans = N）：维度 m×n，x 长度 n，y 长度 m
- `A^T`（转置，trans = T）：x 长度 m，y 长度 n

矩阵 A 采用行主序（row-major）存储。

- 对应的接口为：

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const float *A, int lda,
    const float *x, int incx, const float *beta,
    float *y, int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const uint16_t *A, int lda,
    const uint16_t *x, int incx, const float *beta,
    uint16_t *y, int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const uint16_t *A, int lda,
    const uint16_t *x, int incx, const float *beta,
    float *y, int incy, int batchCount);
```

| Param.        | Memory | in/out | 含义 |
| :------------ | :----- | :----: | :--- |
| handle        | Host   | in     | ops-blas 库上下文句柄 |
| trans         | Host   | in     | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T |
| m, n          | Host   | in     | 矩阵 A 的行数 / 列数 |
| alpha         | Host   | in     | 标量乘数 |
| A             | Device | in     | 矩阵 A 数组（batch×m×n 行主序） |
| lda           | Host   | in     | A 矩阵的 leading dimension |
| x             | Device | in     | 向量 x 数组 |
| incx          | Host   | in     | x 向量元素步长 |
| beta          | Host   | in     | 标量乘数 |
| y             | Device | in/out | 向量 y 数组 |
| incy          | Host   | in     | y 向量元素步长 |
| batchCount    | Host   | in     | 批量大小 |

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型</td><td colspan="3" align="center">SgemvBatched / HSHgemvBatched / HSSgemvBatched</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">A</td><td align="center">batch × m × n</td><td align="center">float / uint16_t</td></tr>
  <tr><td align="center">x</td><td align="center">batch × (trans=N: n, trans=T: m)</td><td align="center">float / uint16_t</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">batch × (trans=N: m, trans=T: n)</td><td align="center">float / uint16_t / float</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="3" align="center">gemv_batched</td></tr>
  </table>

- 算子实现：

  trans=N（不转置）：
    - 使用 AIV SIMD 向量指令实现行级点积（VEC_SCOPE），支持 m-tiling 和 n-tiling 分片策略
    - 多核并行：按 batch 数均匀分配到多个 AIV Core

  trans=T（转置）：
    - 使用 SIMT 编程模型，每个线程处理一个输出元素

- 调用实现：
  使用 `gemv_batched_kernel_do()` 封装内核调用。

## 编译运行

- 环境配置
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译算子
  ```bash
  bash build.sh --ops=gemv_batched --soc=ascend950
  ```

- 编译并运行测试
  ```bash
  bash build.sh --ops=gemv_batched --soc=ascend950 --run
  ```
