# GelsBatched算子

## 算子概述

GelsBatched（批量线性最小二乘/最小范数求解）算子对一批矩阵独立求解线性最小二乘或最小范数问题。属于 LAPACK 风格的批量求解算子，基于 Householder 反射实现 QR/LQ 分解。接口签名严格对齐 cuBLAS `cublasSgelsBatched`。

数学表达式：

```
当 trans == ACLBLAS_OP_N 时：
  超定 (m >= n): min || C[i] - A[i] * X ||_2
    → QR 分解: A = Q*R, X = R^{-1} * Q^T * C
  欠定 (m <  n): min || X ||_2, s.t. A[i]*X = C
    → LQ 分解: A = L*Q, X = Q^T * L^{-1} * C

当 trans == ACLBLAS_OP_T 时：
  将 A[i] 替换为 A[i]^T，即求解 A[i]^T * X = C[i] 的最小二乘/最小范数解。
  Host 侧交换 m/n 并设置转置标志，Kernel 内部执行矩阵转置后统一按 OP_N 处理。
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgelsBatched | 单精度批量最小二乘/最小范数求解 |

## 算子执行接口

### aclblasSgelsBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgelsBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *devInfo, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 操作类型：ACLBLAS_OP_N(111) 不转置；ACLBLAS_OP_T(112) 转置。实数类型不支持 ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A[i] 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A[i] 的列数，n >= 0，Host 内存 |
| nrhs | 输入 | int | 右端项个数（C[i] 的列数），nrhs >= 0，Host 内存 |
| Aarray | 输入/输出 | float *const []（FP32） | 设备指针数组，含 batchSize 个指针，每个指向 m x n 的 float 矩阵（列主序）。分解后 A 被覆盖为 QR/LQ 因子，Device 内存 |
| lda | 输入 | int | A[i] 的 leading dimension，lda >= max(1, m)，Host 内存 |
| Carray | 输入/输出 | float *const []（FP32） | 设备指针数组，含 batchSize 个指针，每个指向 max(m,n) x nrhs 的 float 矩阵（列主序）。输入时前 m 行为右端项 b，输出时前 n 行为解 X，Device 内存 |
| ldc | 输入 | int | C[i] 的 leading dimension，ldc >= max(1, m, n)，Host 内存 |
| devInfo | 输出 | int* | 设备整数数组（长度 batchSize）。devInfo[i]=0 表示第 i 批次成功；devInfo[i]>0 表示第 i 批次秩亏损，Device 内存 |
| batchSize | 输入 | int | 批次数量，batchSize >= 0，Host 内存 |

#### 约束说明

- m >= 0, n >= 0, nrhs >= 0, batchSize >= 0
- lda >= max(1, m)
- ldc >= max(1, m, n)
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T
- m==0 或 n==0 或 nrhs==0 或 batchSize==0 时直接返回成功

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <cstdio>

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int nrhs = 1;
    constexpr int lda = m;
    constexpr int ldc = (m > n) ? m : n;
    constexpr int batchSize = 1;

    float hA[lda * n] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f};
    float hC[ldc * nrhs] = {1.0f, 2.0f, 0.0f};
    int hInfo[batchSize] = {0};

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    void *dA = nullptr, *dC = nullptr, *dInfo = nullptr;
    void *dAPtrs = nullptr, *dCPtrs = nullptr;
    aclrtMalloc(&dA, sizeof(hA), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC, sizeof(hC), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dInfo, sizeof(hInfo), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dAPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, sizeof(hA), hA, sizeof(hA), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dC, sizeof(hC), hC, sizeof(hC), ACL_MEMCPY_HOST_TO_DEVICE);

    float *dAPtrHost = static_cast<float*>(dA);
    float *dCPtrHost = static_cast<float*>(dC);
    aclrtMemcpy(dAPtrs, sizeof(float*), &dAPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dCPtrs, sizeof(float*), &dCPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasSgelsBatched(
        handle, ACLBLAS_OP_N, m, n, nrhs,
        reinterpret_cast<float* const*>(dAPtrs), lda,
        reinterpret_cast<float* const*>(dCPtrs), ldc,
        static_cast<int*>(dInfo), batchSize);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hC, sizeof(hC), dC, sizeof(hC), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hInfo, sizeof(hInfo), dInfo, sizeof(hInfo), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("info[0] = %d\n", hInfo[0]);
    for (int i = 0; i < n; i++)
        printf("x[%d] = %f\n", i, hC[i]);

    aclrtFree(dA);
    aclrtFree(dC);
    aclrtFree(dInfo);
    aclrtFree(dAPtrs);
    aclrtFree(dCPtrs);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
