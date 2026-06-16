## Sspr算子实现

## 概述

本样例展示 `aclblasSspr` 在 Ascend A5 平台上的基本使用流程。

Sspr (Symmetric Packed Rank-1 Update) 实现对称矩阵 packed 格式的秩-1更新操作，数学表达式为：

```
A := alpha * x * x^T + A
```

其中 A 为 n×n 对称矩阵，以 packed 列优先格式存储，仅上三角或下三角区域被引用和更新。

## 支持的产品

- Atlas A5 系列产品 (Ascend950)

## 目录结构介绍

```
├── spr
│   ├── README.md                   // 说明文档
│   └── arch35/
│       ├── sspr_host.cpp            // arch35 Host 侧实现
│       ├── sspr_kernel.cpp          // arch35 Kernel 实现（SIMT）
│       └── sspr_tiling_data.h       // Tiling 数据结构定义
```

## 算子描述

- 算子功能：
  Sspr算子实现对称矩阵 packed 格式的秩-1更新操作，将 `alpha * x * x^T` 加到 packed 对称矩阵的指定三角区域。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sspr</td></tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">AP</td><td align="center">n*(n+1)/2</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">scalar</td><td align="center">float32</td><td align="center">-</td></tr>
  <tr><td align="center">incx</td><td align="center">scalar</td><td align="center">int</td><td align="center">-</td></tr>
  <tr><td align="center">uplo</td><td align="center">scalar</td><td align="center">enum</td><td align="center">ACLBLAS_UPPER / ACLBLAS_LOWER</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">AP</td><td align="center">n*(n+1)/2</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sspr_kernel</td></tr>
  </table>

- 调用实现：
  本样例为 Host API 调用示例，使用 `aclblasSspr` 接口完成算子配置与执行。

- 接口定义：

```c
aclblasStatus_t aclblasSspr(aclblasHandle_t handle,
                            aclblasFillMode_t uplo,
                            int n,
                            const float *alpha,
                            const float *x, int incx,
                            float *ap);
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
| ap | packed 对称矩阵指针，长度 `n*(n+1)/2` |

- 返回码说明：

| 返回码 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 执行成功 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空 |
| ACLBLAS_STATUS_INVALID_VALUE | 参数无效（n<0, incx=0, 空指针等） |
| ACLBLAS_STATUS_EXECUTION_FAILED | 执行失败 |

- 精度指标：

| 数据类型 | MERE 阈值 | MARE 上限 |
|---------|----------|----------|
| float32 | 2^-13 | 10 × 2^-13 |

## 关键设计

- 双路径架构：UB-x 快速路径（incx==1 且 128<=n<=8192）将 x 向量缓存到 UB 共享；GM 回退路径处理其他情况
- 多核并行：按列切分，grid-stride loop 负载均衡
- Packed 索引：UPPER `col*(col+1)/2`，LOWER `col*(2n-col+1)/2`，使用 uint64_t 防溢出

## 编译运行

- 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

- 样例执行

```bash
bash build.sh --ops=spr --soc=ascend950 --run
```

## 注意事项

1. 本算子仅支持 Ascend950 (A5) 平台
2. 接口对齐 cuBLAS cublasSspr，alpha 为指针类型 `const float *`
