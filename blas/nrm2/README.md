# Nrm2算子

## 算子概述

向量范数算子，计算向量的欧几里得范数（2-范数），常用于向量长度计算、归一化和误差估计。

数学表达式：

```
result = sqrt(sum(|x[i]|^2)) for i = 0 to n-1
```

复数向量（Scnrm2）：

```
result = sqrt(sum(|z[i]|^2)) = sqrt(sum(real[i]^2 + imag[i]^2))  for i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSnrm2 | 实数向量欧几里得范数 |
| aclblasScnrm2 | 复数向量欧几里得范数（复用 snrm2 kernel） |

## 算子执行接口

### aclblasSnrm2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 向量元素个数，Host 内存 |
| x | 输入 | uint8_t*（FP32） | 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| result | 输出 | uint8_t*（FP32） | 向量的欧几里得范数，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasScnrm2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 复数向量元素个数，Host 内存 |
| x | 输入 | uint8_t*（FP32 complex） | 复数向量，包含 n 个 complex 元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| result | 输出 | uint8_t*（FP32） | 复数向量的欧几里得范数，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。