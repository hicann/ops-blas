# Ger算子

## 算子概述

Ger（Rank-1 Update）算子实现了矩阵的秩-1更新操作。

数学表达式：

```
A = A + alpha * x * y^T
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSger | 单精度浮点矩阵秩-1更新 |

## 算子执行接口

### aclblasSger

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSger(aclblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| x | 输入 | const float*（FP32） | 长度为 m 的列向量，Device 内存 |
| incx | 输入 | int | 向量 x 的步长，Host 内存 |
| y | 输入 | const float*（FP32） | 长度为 n 的列向量，Device 内存 |
| incy | 输入 | int | 向量 y 的步长，Host 内存 |
| A | 输入/输出 | float*（FP32） | m x n 矩阵，原地更新，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- incx != 0, incy != 0
- lda >= max(1, m)
