# Asum算子

## 算子概述

向量运算算子，计算向量元素绝对值之和（L1 范数 / 曼哈顿范数），常用于向量稀疏度度量和误差估计。

数学表达式：

```
result = sum(|x[i]|) for i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSasum | 实数向量绝对值之和 |

## 算子执行接口

### aclblasSasum

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSasum(aclblasHandle_t handle, int n, const float *x, int incx, float *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0，Host 内存 |
| result | 输出 | float*（FP32） | 向量元素绝对值之和，Device 内存 |

#### 约束说明

- n >= 0（n < 0 时返回错误）
- incx != 0

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。