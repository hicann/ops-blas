# Gemv算子

## 算子概述

Gemv（General Matrix-Vector multiplication）算子实现了通用矩阵与向量的乘法运算。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemv | 单精度浮点矩阵-向量乘法 |
| aclblasCgemv | 复数矩阵-向量乘法 |

## 算子执行接口

### aclblasSgemv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数等价于转置），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 列主序 m x n 矩阵，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，lda >= max(1, m)，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，trans=N 时逻辑长度 n，trans=T/C 时逻辑长度 m，Device 内存 |
| incx | 输入 | int | 向量 x 的元素步长，incx != 0，支持正负值，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta，不可为 nullptr。若 beta == 0，则 y 的输入值不被使用，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，trans=N 时逻辑长度 m，trans=T/C 时逻辑长度 n，Device 内存 |
| incy | 输入 | int | 向量 y 的元素步长，incy != 0，支持正负值，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- alpha、beta 不可为 nullptr

### aclblasCgemv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemv(aclblasHandle_t handle, aclblasOperation trans, const int64_t m, const int64_t n, const std::complex<float>& alpha, uint8_t* A, const int64_t lda, uint8_t* x, const int64_t incx, const std::complex<float>& beta, uint8_t* y, const int64_t incy);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型：N=不转置，T=转置，C=共轭转置，Host 内存 |
| m | 输入 | int64_t | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int64_t | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const std::complex<float>&（FP32 complex） | 复数标量 alpha，Host 内存 |
| A | 输入 | uint8_t* | m x n 复数矩阵，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维长度，Host 内存 |
| x | 输入 | uint8_t* | 向量 x（长度取决于 trans），Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| beta | 输入 | const std::complex<float>&（FP32 complex） | 复数标量 beta，Host 内存 |
| y | 输入/输出 | uint8_t* | 向量 y（长度取决于 trans），Device 内存 |
| incy | 输入 | int64_t | y 中连续元素之间的步长，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
