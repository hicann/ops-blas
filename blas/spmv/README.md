# Spmv算子

## 算子概述

Spmv（Symmetric Packed Matrix-Vector Multiplication）算子实现了对称压缩矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行了优化，采用压缩存储格式以节省内存空间，并高效完成矩阵与向量的乘加运算。

数学表达式：

```
z = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSpmv | 单精度浮点对称压缩矩阵-向量乘法 |
| aclblasSspmv | 单精度浮点对称压缩矩阵-向量乘法 |

## 算子执行接口

### aclblasSpmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSpmv(aclblasHandle_t handle, const float *aPacked, const float *x, const float *y, float *z, const float alpha, const float beta, const int64_t n, const int64_t incx, const int64_t incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| aPacked | 输入 | const float*（FP32） | 对称压缩矩阵，n*(n+1)/2 个元素，Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量，包含 n 个元素，Device 内存 |
| y | 输入 | const float*（FP32） | 输入向量，包含 n 个元素，Device 内存 |
| z | 输出 | float*（FP32） | 输出向量，包含 n 个元素，Device 内存 |
| alpha | 输入 | float | 标量乘数，Host 内存 |
| beta | 输入 | float | 标量乘数，Host 内存 |
| n | 输入 | int64_t | 对称压缩矩阵 A 的行数和列数，Host 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| incy | 输入 | int64_t | y 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0, incy != 0
### aclblasSspmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSspmv(aclblasHandle_t handle, aclblasFillMode uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 的指针，Host 内存 |
| AP | 输入 | const float*（FP32） | 对称压缩矩阵，共 n(n+1)/2 个元素，Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量，n 个元素，Device 内存 |
| incx | 输入 | int | x 的步长，incx != 0（可正可负），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 的指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，n 个元素，Device 内存 |
| incy | 输入 | int | y 的步长，incy != 0（可正可负），Host 内存 |

#### 约束说明

- n >= 0
- incx != 0, incy != 0