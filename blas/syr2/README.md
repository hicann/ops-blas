# Syr2算子

## 算子概述

syr2 (Symmetric Rank-2 Update) 实现对称秩-2更新操作。该算子将两个向量的外积组合加到对称矩阵的指定三角区域。

数学表达式：

```
A = alpha * x * y^T + alpha * y * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyr2 | 单精度对称秩-2更新 |

## 算子执行接口

### aclblasSsyr2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyr2(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 的存储格式。ACLBLAS_LOWER(122): 下三角，ACLBLAS_UPPER(121): 上三角，Host 内存 |
| n | 输入 | int | 向量 x 和 y 中的元素个数，矩阵 A 的行列数。n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 指针，向量乘积缩放因子，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，对应公式中的 x。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n]，Device 内存 |
| incx | 输入 | int | x 相邻元素间的内存地址偏移量，incx != 0，Host 内存 |
| y | 输入 | const float*（FP32） | 输入向量，对应公式中的 y。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n]，Device 内存 |
| incy | 输入 | int | y 相邻元素间的内存地址偏移量，incy != 0，Host 内存 |
| A | 输入/输出 | float*（FP32） | 输入/输出矩阵，对应公式中的 A。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n, n]，Device 内存 |
| lda | 输入 | int | 矩阵 A 的每列元素的存储步长，lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0，n==0 时直接返回成功
- incx != 0，incy != 0
- lda >= max(1, n)
- 算子输入 shape 为 [n]、[n]、[n, n]，输出 shape 为 [n, n]
- 算子实际计算时，不支持 ND 高维度运算（不支持维度 >= 3 的运算）
- Host 侧不做流同步，调用方需自行管理同步
