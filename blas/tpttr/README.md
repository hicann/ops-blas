# Tpttr算子

## 算子概述

Tpttr（Symmetric Triangular matrix, Packed format To Triangular matrix, Regular storage）算子将 LAPACK 压缩格式（packed format）中的对称三角矩阵展开为按列主序存储的常规二维矩阵。仅写入 `uplo` 指定的三角区域，矩阵另一三角及未参与运算的元素保持原值不变。属于 LAPACK 格式转换算子。

数学表达式：

```
A[triangular] = unpack(AP)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpttr | 单精度压缩三角矩阵展开为常规矩阵 |

## 算子执行接口

### aclblasStpttr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpttr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵维数，须 >= 0；为 0 时立即返回成功，Host 内存 |
| AP | 输入 | const float*（FP32） | 压缩格式输入，float 数组，长度 n*(n+1)/2，Device 内存 |
| A | 输入/输出 | float*（FP32） | 常规输出矩阵，float 数组，维度 lda × n；非目标三角保持原值，Device 内存 |
| lda | 输入 | int | A 的主维长度，须满足 lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0
- lda >= max(1, n)
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。