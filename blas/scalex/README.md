# Scalex算子

## 算子概述

Scalex（混合精度向量标量乘）算子实现 `x[j] = alpha * x[j]`，其中 `j = (i - 1) * incx`，`i = 1, 2, ..., n`，支持混合精度计算。

数学表达式：

```
x[j] = alpha * x[j], j = (i - 1) * incx, i = 1, 2, ..., n
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasScalex | 混合精度向量标量乘（alpha=FP32, x in {FP16/BF16/FP32}, execution=FP32） |

## 算子执行接口

### aclblasScalex

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScalex(aclblasHandle_t handle, int n, const void *alpha, aclDataType alphaType, void *x, aclDataType xType, int incx, aclDataType executionType)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| alpha | 输入 | const void* | 标量因子指针，实际类型由 alphaType 指定，Host/Device 内存 |
| alphaType | 输入 | aclDataType | alpha 数据类型，固定为 ACL_FLOAT，Host 内存 |
| x | 输入/输出 | void* | Device 端向量指针，类型由 xType 指定，Device 内存 |
| xType | 输入 | aclDataType | 向量 x 的数据类型，Host 内存 |
| incx | 输入 | int | x 中相邻元素的步长，Host 内存 |
| executionType | 输入 | aclDataType | 计算精度类型，固定为 ACL_FLOAT，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- alphaType 固定为 ACL_FLOAT
- executionType 固定为 ACL_FLOAT
- xType 必须为 ACL_FLOAT、ACL_FLOAT16 或 ACL_BF16