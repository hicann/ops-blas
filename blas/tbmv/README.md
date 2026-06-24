# Tbmv算子

## 算子概述

tbmv (Triangular Band Matrix-Vector Multiplication) 实现三角带状矩阵与向量的乘法运算。该算子支持上三角和下三角矩阵，支持转置和共轭转置操作，支持单位对角线和非单位对角线。

数学表达式：

```
x = op(A) * x    （arch35，原地覆盖）
y = A * x        （arch22，输入输出分离）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStbmv | 单精度三角带状矩阵-向量乘法（标准接口） |
| aclblasStbmv_legacy | 单精度三角带状矩阵-向量乘法（早期接口） |

## 算子执行接口

### aclblasStbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同 T)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为 1)，Host 内存 |
| n | 输入 | int | 三角带状矩阵 A 的行数和列数，Host 内存 |
| k | 输入 | int | 三角带状矩阵的半带宽，Host 内存 |
| A | 输入 | const float*（FP32） | 三角带状矩阵 float 数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 矩阵 A 存储的主维长度，lda >= k + 1，Host 内存 |
| x | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- lda >= k + 1
- incx != 0

### aclblasStbmv_legacy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbmv_legacy(aclblasHandle_t handle, const float *a, const int64_t lda, const float *x, float *y, const int64_t n, const int64_t k, const int64_t incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| a | 输入 | const float*（FP32） | 下三角带状矩阵 float 数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int64_t | 矩阵 a 存储的主维长度，lda >= k + 1，Host 内存 |
| x | 输入 | const float*（FP32） | float 输入向量，包含 n 个元素，Device 内存 |
| y | 输出 | float*（FP32） | float 输出向量，包含 n 个元素，Device 内存 |
| n | 输入 | int64_t | 矩阵 A 的行数和列数，Host 内存 |
| k | 输入 | int64_t | 下三角带状矩阵的半带宽，Host 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- lda >= k + 1
- incx != 0
