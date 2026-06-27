# Iamin算子

## 算子概述

Iamin 算子实现了查找向量中绝对值最小元素的索引，核心运算为遍历向量取绝对值并比较大小。该算子返回 1-based 索引，遵循 BLAS 惯例，常用于稀疏矩阵预处理和数值优化中。

数学表达式：

```
result = argmin_i |x[i]|
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasIsamin | 查找 FP32 向量中绝对值最小元素的 1-based 索引 |

## 算子执行接口

### aclblasIsamin

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float *x, int incx, int *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const float* | 指向 float 向量的 device 指针，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻元素之间的步长，Host 内存 |
| result | 输出 | int* | 绝对值最小元素的 1-based 索引，Device 内存 |

#### 约束说明

- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE
- n = 0 或 incx < 1 时 result 写 0，返回 ACLBLAS_STATUS_SUCCESS
- handle 不可为 nullptr
- x、result 不可为 nullptr
- 当多个元素绝对值相同时，返回索引最小的元素
