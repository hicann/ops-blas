# Gemm算子

## 算子概述

通用矩阵乘法扩展接口（GEMM Ex），支持 A、B、C 矩阵使用独立数据类型。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasGemmEx | 通用矩阵乘法扩展接口 |

## 算子执行接口

### aclblasGemmEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasGemmEx(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb, const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transA），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha，指向 float 类型的指针，Host 内存 |
| A | 输入 | const void* | 矩阵 A 的设备内存指针，Device 内存 |
| Atype | 输入 | aclDataType | 矩阵 A 的数据类型，Host 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），Host 内存 |
| B | 输入 | const void* | 矩阵 B 的设备内存指针，Device 内存 |
| Btype | 输入 | aclDataType | 矩阵 B 的数据类型，Host 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），Host 内存 |
| beta | 输入 | const void* | 标量 beta，指向 float 类型的指针，Host 内存 |
| C | 输入/输出 | void* | 矩阵 C 的设备内存指针，Device 内存 |
| Ctype | 输入 | aclDataType | 矩阵 C 的数据类型，Host 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, m)，Host 内存 |
| computeType | 输入 | aclblasComputeType_t | 计算精度类型，Host 内存 |
| algo | 输入 | aclblasGemmAlgo_t | 算法选择，当前仅支持 ACLBLAS_GEMM_DEFAULT，Host 内存 |

#### 约束说明

- m, n, k >= 0
- ldc >= max(1, m)
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- algo 当前仅支持 ACLBLAS_GEMM_DEFAULT
