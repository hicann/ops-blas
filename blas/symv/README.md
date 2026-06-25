# Symv算子

## 算子概述

symv (Symmetric Matrix-Vector Multiplication) 实现对称矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行优化，仅存储上三角或下三角部分，通过对称性推断未存储部分，高效完成矩阵与向量的乘加运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymv | 单精度对称矩阵-向量乘法 |

## 算子执行接口

### aclblasSsymv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymv(aclblasHandle_t handle, aclblasFillMode uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | 指定矩阵 A 存储上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER），Host 内存 |
| n | 输入 | int | 对称矩阵 A 的行数和列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 用于乘法的 float 标量，Host/Device 内存 |
| A | 输入 | const float*（FP32） | 对称矩阵 float 数组，维度为 lda x n，仅存储 uplo 指定的三角部分，Device 内存 |
| lda | 输入 | int | 用于存储矩阵 A 的二维数组的主维，lda >= max(1, n)，Host 内存 |
| x | 输入 | const float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，incx != 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 用于乘法的 float 标量。如果 beta == 0，则 y 不必是有效输入，Host/Device 内存 |
| y | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素。输入为初始 y 值，输出为计算结果，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，incy != 0，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0
- lda >= max(1, n)

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。