# Tpmv算子

## 算子概述

tpmv (Triangular Packed Matrix-Vector Multiplication) 实现三角压缩矩阵与向量的乘法运算。该算子针对三角矩阵的 packed 存储特性进行优化，采用压缩存储格式以节省内存空间，高效完成矩阵与向量的乘加运算。

数学表达式：

```
x = A * x
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpmv | 单精度三角压缩矩阵-向量乘法（标准接口） |
| aclblasStpmv_legacy | 单精度三角压缩矩阵-向量乘法（早期接口） |

## 算子执行接口

### aclblasStpmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpmv(aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n, const float *aPacked, const float *x, float *y, int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线)，Host 内存 |
| n | 输入 | int64_t | 三角压缩矩阵的行数和列数，Host 内存 |
| aPacked | 输入 | const float*（FP32） | 三角压缩矩阵 float 数组，维度为 n*(n+1)/2，Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量，包含 n 个元素，Device 内存 |
| y | 输出 | float*（FP32） | 输出向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasStpmv_legacy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpmv_legacy(aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n, const float *aPacked, const float *x, float *y, int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | 矩阵填充类型：ACLBLAS_UPPER 或 ACLBLAS_LOWER，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型，Host 内存 |
| diag | 输入 | aclblasDiagType | 对角线类型，Host 内存 |
| n | 输入 | int64_t | 三角压缩矩阵 A 的行数和列数，Host 内存 |
| aPacked | 输入 | const float*（FP32） | 三角压缩矩阵 float 数组，Device 内存 |
| x | 输入 | const float*（FP32） | float 输入向量，包含 n 个元素，Device 内存 |
| y | 输出 | float*（FP32） | float 输出向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。