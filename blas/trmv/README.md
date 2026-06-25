# Trmv算子

## 算子概述

trmv (Triangular Matrix-Vector Multiplication) 实现三角矩阵与向量的乘法运算。本算子包含实数三角矩阵-向量乘法（Strmv）和复数三角矩阵-向量乘法（Ctrmv）。

数学表达式：

```
x = op(A) * x
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrmv | 实数三角矩阵-向量乘法 |
| aclblasCtrmv | 复数三角矩阵-向量乘法 |

## 算子执行接口

### aclblasStrmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrmv(aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, const int64_t n, uint8_t *A, const int64_t lda, uint8_t *x, const int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同 T)，Host 内存 |
| diag | 输入 | aclblasDiagType | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为 1)，Host 内存 |
| n | 输入 | int64_t | 三角矩阵 A 的行数和列数，Host 内存 |
| A | 输入 | uint8_t* | 三角矩阵数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 存储的主维长度，lda >= n，Host 内存 |
| x | 输入/输出 | uint8_t* | 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖），Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，不可为 0，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- lda >= n

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasCtrmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n, uint8_t *A, int64_t lda, uint8_t *x, int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 的上三角或下三角部分。ACLBLAS_UPPER 或 ACLBLAS_LOWER，Host 内存 |
| trans | 输入 | aclblasOperation_t | 指定对矩阵 A 的操作类型。ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 指定对角线元素是否为单位元。ACLBLAS_UNIT(单位对角线)或 ACLBLAS_NON_UNIT(非单位对角线)，Host 内存 |
| n | 输入 | int64_t | 矩阵 A 的阶数，即向量的长度，Host 内存 |
| A | 输入 | uint8_t* | n x lda 的复数矩阵，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维度，Host 内存 |
| x | 输入/输出 | uint8_t* | 复数向量，长度为 n。既是输入也是输出，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n 的取值范围为 [1, 8192]
- 仅支持 complex<float> 数据类型
- incx > 0
- lda > 0

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。