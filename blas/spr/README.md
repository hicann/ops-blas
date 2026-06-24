# Spr算子

## 算子概述

spr (Symmetric Packed Rank-1 Update) 实现对称矩阵 packed 格式的秩-1更新操作。该算子将 `alpha * x * x^T` 加到 packed 对称矩阵的指定三角区域。

数学表达式：

```
A := alpha * x * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSspr | 单精度对称 packed 秩-1更新 |

## 算子执行接口

### aclblasSspr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSspr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *ap);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数指针，Host/Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量指针，长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0，Host 内存 |
| ap | 输入/输出 | float*（FP32） | packed 对称矩阵指针，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0
