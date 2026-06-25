# GeqrfBatched算子

## 算子概述

GeqrfBatched（批量 QR 分解）算子对一批矩阵使用 Householder 反射进行 QR 分解。对每个批次 j（j = 0, 1, ..., batchSize-1），将 m x n 实矩阵 A[j] 分解为 A[j] = Q[j] * R[j]。属于 LAPACK 风格的批量分解算子。

数学表达式：

```
A[j] = Q[j] * R[j],   j = 0, 1, ..., batchSize-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgeqrfBatched | 单精度批量 QR 分解 |

## 算子执行接口

### aclblasSgeqrfBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgeqrfBatched(aclblasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| m | 输入 | int | 每个矩阵 Aarray[i] 的行数，要求 m >= 0，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的列数，要求 n >= 0，Host 内存 |
| Aarray | 输入/输出 | float *const []（FP32） | 设备端指针数组，每个元素指向一个 m x n 列主序矩阵。输入时包含原始矩阵，输出时下三角存储 Householder 向量 v，上三角存储 R，Device 内存 |
| lda | 输入 | int | 矩阵 Aarray[i] 的前导维度，要求 lda >= max(1, m)，Host 内存 |
| TauArray | 输出 | float *const []（FP32） | 设备端指针数组，每个元素指向维度 min(m, n) 的向量，存储 Householder 标量因子 tau，Device 内存 |
| info | 输出 | int* | Host 端指针，指向单个 int。0 = 成功，Host 内存 |
| batchSize | 输入 | int | Aarray 中包含的指针数量（批次数），要求 batchSize >= 0，Host 内存 |

#### 约束说明

- m >= 0, n >= 0, batchSize >= 0
- lda >= max(1, m)