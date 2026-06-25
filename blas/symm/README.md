# Symm算子

## 算子概述

Symm（Single-precision Symmetric Matrix Multiplication）算子实现了单精度浮点对称矩阵与普通矩阵的乘法运算。

数学表达式：

```
LEFT 模式：C := alpha * A * B + beta * C
RIGHT 模式：C := alpha * B * A + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymm | 单精度浮点对称矩阵乘法 |

## 算子执行接口

### aclblasSsymm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymm(aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_LOWER（下三角）或 ACLBLAS_UPPER（上三角），Host 内存 |
| m | 输入 | int64_t | 矩阵 C 的行数，m >= 0，Host 内存 |
| n | 输入 | int64_t | 矩阵 C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 对称矩阵，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维，side=LEFT 时 lda >= max(1, m)，side=RIGHT 时 lda >= max(1, n)，Host 内存 |
| B | 输入 | const float*（FP32） | m×n 普通矩阵，Device 内存 |
| ldb | 输入 | int64_t | 矩阵 B 的主维，ldb >= max(1, n)，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta，不可为 nullptr，Host 内存 |
| C | 输入/输出 | float*（FP32） | m×n 矩阵，输入旧值，输出新值，Device 内存 |
| ldc | 输入 | int64_t | 矩阵 C 的主维，ldc >= max(1, n)，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- side=LEFT 时：lda >= max(1, m)
- side=RIGHT 时：lda >= max(1, n)
- ldb >= max(1, n)
- ldc >= max(1, n)
- alpha、beta 不可为 nullptr