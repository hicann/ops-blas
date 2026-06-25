# GetrsBatched算子

## 算子概述

GetrsBatched（批量线性方程组求解）算子对一批已经由 aclblasSgetrfBatched 完成 LU 分解的 n×n 方阵，批量求解线性方程组。属于 LAPACK 风格的批量求解算子，接口对齐 LAPACK sgetrs 标准。

数学表达式：

```
op(A[i]) * X[i] = B[i],   i = 0, 1, ..., batchCount - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetrsBatched | 单精度批量线性方程组求解 |

## 算子执行接口

### aclblasSgetrsBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetrsBatched(aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 转置操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数域等价于转置），Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| nrhs | 输入 | int | 每个矩阵 Barray[i] 的列数（右端项数量），nrhs >= 0，nrhs <= 256，Host 内存 |
| Aarray | 输入 | const float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中已 LU 分解的 n×n float 矩阵（列主序），Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| devIpiv | 输入 | const int* | 大小为 n × batchCount 的数组，存储每个矩阵的主元序列（来自 aclblasSgetrfBatched 输出），可为 NULL，Device 内存 |
| Barray | 输入/输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n×nrhs float 矩阵（列主序），输入时为右端矩阵 B，输出时被覆盖为解矩阵 X，Device 内存 |
| ldb | 输入 | int | 每个矩阵 Barray[i] 的 leading dimension，ldb >= max(1, n)，Host 内存 |
| info | 输出 | int* | 仅反映参数级错误，*info == 0 表示参数校验通过；*info = -j 表示第 j 个参数非法，Host 内存 |
| batchCount | 输入 | int | 指针数组中包含的矩阵数量，batchCount >= 0，Host 内存 |

#### 约束说明

- n >= 0, nrhs >= 0, batchCount >= 0
- nrhs <= 256
- lda >= max(1, n), ldb >= max(1, n)
- n == 0 或 nrhs == 0 或 batchCount == 0 时直接返回成功，不启动 Kernel
- 矩阵以列主序（Column-major）存储，与 LAPACK 标准一致
- 调用前必须先使用 aclblasSgetrfBatched 对 Aarray[i] 完成 LU 分解

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。