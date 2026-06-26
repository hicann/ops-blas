# Scal算子

## 算子概述

向量缩放算子，实现向量乘以标量的运算。包含实数向量缩放（Sscal）和复数向量缩放（Cscal）。

数学表达式：

```
x[i] = alpha * x[i]  (i = 0 .. n-1，步长为 incx)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSscal | 实数向量乘以标量 |
| aclblasCscal | 复数向量乘以复数标量 |

## 算子执行接口

### aclblasSscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 中的元素个数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量乘数的指针，Host 内存 |
| x | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

**Ascend 950PR / Ascend 950DT（arch35）：**

- n 为整数；n <= 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不修改 x）
- incx 为整数；incx <= 0 时为 no-op（不修改 x，对齐参考 BLAS cblas_sscal 的 IF (INCX.LE.0) RETURN 语义）；incx > 0 时支持任意步长
- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n > 0 时 alpha、x 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE

**Atlas A2 / Atlas A3 系列产品（arch22）：**

- incx 参数当前实现未启用（固定按连续向量 incx=1 处理，传入的 incx 取值不生效）
- 未对 n、handle、alpha、x 做入参校验

### aclblasCscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCscal(aclblasHandle_t handle, const int64_t n, const std::complex<float> alpha, uint8_t* x, const int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 向量 x 中的复数元素个数，Host 内存 |
| alpha | 输入 | const std::complex<float> | 用于乘法的复数标量，Host 内存 |
| x | 输入/输出 | uint8_t*（FP32 complex） | 复数向量，包含 n 个 complex<float> 元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0