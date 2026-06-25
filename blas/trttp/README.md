# Trttp算子

## 算子概述

Trttp（Triangular matrix, Regular storage To Triangular matrix, Packed format）算子将常规二维三角矩阵压缩为 packed format 存储。属于 LAPACK 格式转换算子。

数学表达式：

```
A(lda x n) -> AP(n*(n+1)/2, packed format)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrttp | 单精度常规三角矩阵压缩为 packed 格式 |

## 算子执行接口

### aclblasStrttp

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *A, int lda, float *AP)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵维数，Host 内存 |
| A | 输入 | const float*（FP32） | 常规三角矩阵，维度 lda × n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda >= max(1, n)，Host 内存 |
| AP | 输出 | float*（FP32） | 输出压缩格式，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0
- lda >= max(1, n)
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER