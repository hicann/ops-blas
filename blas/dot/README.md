# Dot算子

## 算子概述

向量点积算子，实现两个向量的点积运算。包含实数点积（Sdot）和复数点积（Cdot）两类接口。广泛应用于信号处理、统计学、量子计算和线性代数等领域。

数学表达式：

```
result = x · y = Σ(x[i] * y[i])  for i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSdot | 实数向量点积 |
| aclblasCdotu | 无共轭复数点积 |
| aclblasCdotc | 共轭复数点积 |

## 算子执行接口

### aclblasSdot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSdot(aclblasHandle_t handle, const int64_t n, const float *x, const int64_t incx, const float *y, const int64_t incy, float *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 实数向量的元素个数，Host 内存 |
| x | 输入 | const float*（FP32） | 实数向量，包含 n 个 float 元素，Device 内存 |
| incx | 输入 | int64_t | 向量 x 的步长，不可为 0，Host 内存 |
| y | 输入 | const float*（FP32） | 实数向量，包含 n 个 float 元素，Device 内存 |
| incy | 输入 | int64_t | 向量 y 的步长，不可为 0，Host 内存 |
| result | 输出 | float*（FP32） | 实数结果，包含 1 个 float 元素，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0

### aclblasCdotu

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasCdotu(const float *x, const float *y, float *result, const int64_t n, void *stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| x | 输入 | const float*（FP32） | 复数向量，包含 2*n 个 float 元素（实部和虚部交替存储），Device 内存 |
| y | 输入 | const float*（FP32） | 复数向量，包含 2*n 个 float 元素（实部和虚部交替存储），Device 内存 |
| result | 输出 | float*（FP32） | 复数结果，包含 2 个 float 元素（实部和虚部），Device 内存 |
| n | 输入 | int64_t | 复数向量的元素个数，Host 内存 |
| stream | 输入 | void* | ACL stream，Host 内存 |

#### 约束说明

- n >= 0

### aclblasCdotc

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasCdotc(const float *x, const float *y, float *result, const int64_t n, void *stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| x | 输入 | const float*（FP32） | 复数向量，包含 2*n 个 float 元素（实部和虚部交替存储），Device 内存 |
| y | 输入 | const float*（FP32） | 复数向量，包含 2*n 个 float 元素（实部和虚部交替存储），Device 内存 |
| result | 输出 | float*（FP32） | 复数结果，包含 2 个 float 元素（实部和虚部），Device 内存 |
| n | 输入 | int64_t | 复数向量的元素个数，Host 内存 |
| stream | 输入 | void* | ACL stream，Host 内存 |

#### 约束说明

- n >= 0