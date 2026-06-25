# Swap算子

## 算子概述

Swap 算子实现了两个向量对应元素的交换操作，属于纯数据搬运类算子，不涉及任何数值计算。

数学表达式：

```
x <-> y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSswap | 实数向量交换 |
| aclblasCswap | 复数向量交换 |

## 算子执行接口

### aclblasSswap

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSswap(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | const int64_t | 向量中参与交换的元素个数，Host 内存 |
| x | 输入/输出 | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 y 的元素，Device 内存 |
| incx | 输入 | const int64_t | 向量 x 中相邻元素之间的步长，Host 内存 |
| y | 输入/输出 | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 x 的元素，Device 内存 |
| incy | 输入 | const int64_t | 向量 y 中相邻元素之间的步长，Host 内存 |

#### 约束说明

- n <= 0 时直接返回成功，不执行任何操作
- handle 不可为 nullptr
- x、y 不可为 nullptr
- incx != 0, incy != 0
### aclblasCswap

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCswap(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | const int64_t | 向量中的复数元素个数，Host 内存 |
| x | 输入/输出 | uint8_t* | 指向复数向量的 device 指针，交换后包含原 y 的元素，Device 内存 |
| incx | 输入 | const int64_t | x 中连续元素之间的步长，Host 内存 |
| y | 输入/输出 | uint8_t* | 指向复数向量的 device 指针，交换后包含原 x 的元素，Device 内存 |
| incy | 输入 | const int64_t | y 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n <= 0 时直接返回成功，不执行任何操作
- incx != 0, incy != 0