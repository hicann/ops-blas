# Sbmv算子

## 算子概述

Sbmv（Symmetric Banded Matrix-Vector Multiplication）算子实现了对称带状矩阵与向量的乘法运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsbmv | 单精度浮点对称带状矩阵-向量乘法 |

## 算子执行接口

### aclblasSsbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsbmv(aclblasHandle_t handle, aclblasFillMode uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵阶数，Host 内存 |
| k | 输入 | int | 次对角线/超对角线数量，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 的指针，Host 内存 |
| A | 输入 | const float*（FP32） | 带状对称矩阵，列主序，维度 (k+1)×n，Device 内存 |
| lda | 输入 | int | A 的主维数，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，n 个元素，Device 内存 |
| incx | 输入 | int | x 的步长（可正可负），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 的指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，n 个元素，Device 内存 |
| incy | 输入 | int | y 的步长（可正可负），Host 内存 |

#### 约束说明

- n >= 0, k >= 0
- lda >= k + 1
- incx != 0, incy != 0
