# Rot算子

## 算子概述

向量旋转算子，实现对两个向量的平面旋转（Givens 旋转），常用于 QR 分解、求解线性方程组和特征值计算等数值算法中。

数学表达式：

```
x[i] = c * x[i] + s * y[i]
y[i] = c * y[i] - s * x[i] (使用原始 x[i])
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCsrot | 复数向量平面旋转 |

## 算子执行接口

### aclblasCsrot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsrot(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy, const float c, const float s)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 向量元素个数，Host 内存 |
| x | 输入/输出 | uint8_t*（FP32 complex） | 向量，包含 n 个元素，原地修改，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| y | 输入/输出 | uint8_t*（FP32 complex） | 向量，包含 n 个元素，原地修改，Device 内存 |
| incy | 输入 | int64_t | y 中连续元素之间的步长，Host 内存 |
| c | 输入 | float | 旋转角度的余弦值，Host 内存 |
| s | 输入 | float | 旋转角度的正弦值，Host 内存 |

#### 约束说明

- n >= 0

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。