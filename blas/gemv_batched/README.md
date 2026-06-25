# GemvBatched算子

## 算子概述

GemvBatched（批量矩阵-向量乘法）实现了对一批矩阵分别进行矩阵-向量乘法的运算。

数学表达式：

```
y[i] = alpha * op(A[i]) * x[i] + beta * y[i]
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemvBatched | 单精度批量矩阵-向量乘法 |
| aclblasHSHgemvBatched | FP16 入/出批量矩阵-向量乘法 |
| aclblasHSSgemvBatched | FP16 入/FP32 出批量矩阵-向量乘法 |
| aclblasCgemvBatched | 复数批量矩阵-向量乘法 |

## 算子执行接口

### aclblasSgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| A | 输入 | const float*（FP32） | 矩阵 A 数组，Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| x | 输入 | const float*（FP32） | 向量 x 数组，Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| y | 输入/输出 | float*（FP32） | 向量 y 数组，Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasHSHgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSHgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, const uint16_t *x, int incx, const float *beta, uint16_t *y, int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| A | 输入 | const uint16_t* | 矩阵 A 数组（FP16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| x | 输入 | const uint16_t* | 向量 x 数组（FP16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| y | 输入/输出 | uint16_t* | 向量 y 数组（FP16），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasHSSgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, const uint16_t *x, int incx, const float *beta, float *y, int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| A | 输入 | const uint16_t* | 矩阵 A 数组（FP16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| x | 输入 | const uint16_t* | 向量 x 数组（FP16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| y | 输入/输出 | float*（FP32） | 向量 y 数组，Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasCgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemvBatched(aclblasHandle_t handle, aclblasOperation trans, const int64_t m, const int64_t n, const std::complex<float>& alpha, uint8_t* A, const int64_t lda, uint8_t* x, const int64_t incx, const std::complex<float>& beta, uint8_t* y, const int64_t incy, const int64_t batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型：N=不转置，T=转置，C=共轭转置，Host 内存 |
| m | 输入 | int64_t | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int64_t | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const std::complex<float>&（FP32 complex） | 复数标量 alpha，Host 内存 |
| A | 输入 | uint8_t* | 批量复数矩阵，batchCount 个 m x n 矩阵，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维长度，Host 内存 |
| x | 输入 | uint8_t* | 批量复数向量，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| beta | 输入 | const std::complex<float>&（FP32 complex） | 复数标量 beta，Host 内存 |
| y | 输入/输出 | uint8_t* | 批量复数向量，Device 内存 |
| incy | 输入 | int64_t | y 中连续元素之间的步长，Host 内存 |
| batchCount | 输入 | int64_t | 批次数，Host 内存 |

#### 约束说明

- batchCount >= 0, m >= 0, n >= 0
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。