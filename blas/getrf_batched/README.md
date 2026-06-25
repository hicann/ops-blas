# GetrfBatched算子

## 算子概述

GetrfBatched（批量 LU 分解）算子对一批 n x n 方阵分别执行带部分主元选取的 LU 分解（LU factorization with partial pivoting）。属于 LAPACK 风格的批量分解算子，接口对齐 cuBLAS `cublasSgetrfBatched`，算法参考 NETLIB LAPACK sgetrf。

数学表达式：

```
P * A[i] = L * U,   i = 0, 1, ..., batchSize - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetrfBatched | 单精度批量 LU 分解（带部分主元选取） |

## 算子执行接口

### aclblasSgetrfBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetrfBatched(aclblasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| Aarray | 输入/输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n x n float 矩阵，矩阵以列主序存储，Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| PivotArray | 输出 | int* | 大小为 n x batchSize 的数组，存储每个矩阵的主元序列（1-indexed，LAPACK 约定），可为 NULL，Device 内存 |
| infoArray | 输出 | int* | 大小为 batchSize 的数组，infoArray[i] = 0 表示分解成功；= k > 0 表示 U(k,k) == 0，Device 内存 |
| batchSize | 输入 | int | 指针数组中包含的矩阵数量，batchSize >= 0，Host 内存 |

#### 约束说明

- n >= 0, batchSize >= 0
- lda >= max(1, n)
- n == 0 或 batchSize == 0 时直接返回成功，不启动 Kernel
- PivotArray != NULL 时 infoArray 不可为 NULL
- PivotArray == NULL 合法（禁用主元选取，执行非主元 LU 分解）

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。