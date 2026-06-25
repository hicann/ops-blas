# ColwiseMul算子

## 算子概述

ColwiseMul（列方向复数乘法）算子实现了复数向量与矩阵每行的逐元素乘法运算，是 BLAS 基础线性代数库中的扩展算子之一。该算子针对复数运算特性进行了优化，高效完成向量与矩阵的列方向乘法操作。

数学表达式：

```
result[i, j] = vec[i] * mat[i, j]
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasColwiseMul | 复数向量与矩阵每行逐元素乘法 |

## 算子执行接口

### aclblasColwiseMul

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasColwiseMul(const float *mat, const float *vec, float *result, const int64_t m, const int64_t n, void *stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| mat | 输入 | const float*（FP32） | 复数矩阵，维度为 m x n，存储为 2*m*n 个 float，Device 内存 |
| vec | 输入 | const float*（FP32） | 复数向量，包含 m 个元素，存储为 2*m 个 float，Device 内存 |
| result | 输出 | float*（FP32） | 复数矩阵，维度为 m x n，存储为 2*m*n 个 float，Device 内存 |
| m | 输入 | int64_t | 矩阵的行数，Host 内存 |
| n | 输入 | int64_t | 矩阵的列数，Host 内存 |
| stream | 输入 | void* | 执行流，Host 内存 |

#### 约束说明

无