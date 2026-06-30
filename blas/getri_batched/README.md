# GetriBatched算子

## 算子概述

GetriBatched（批量矩阵求逆）算子对一批已经由 aclblasSgetrfBatched 完成 LU 分解的 n x n 方阵，批量计算逆矩阵。属于 LAPACK 风格的批量求逆算子，接口对齐 LAPACK sgetri 标准。

数学表达式：

```
inv(A[i]) = inv(U[i]) * inv(L[i]) * P[i],   i = 0, 1, ..., batchSize - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetriBatched | 单精度批量矩阵求逆 |

## 算子执行接口

### aclblasSgetriBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetriBatched(aclblasHandle_t handle, int n, const float *const Aarray[], int lda, const int *PivotArray, float *const Carray[], int ldc, int *infoArray, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| Aarray | 输入 | const float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中已 LU 分解的 n x n float 矩阵（列主序），Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| PivotArray | 输入 | const int* | 大小为 n x batchSize 的数组，存储每个矩阵的主元序列（来自 aclblasSgetrfBatched 输出），可为 NULL，Device 内存 |
| Carray | 输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n x n float 矩阵（列主序），用于存储逆矩阵，Device 内存 |
| ldc | 输入 | int | 每个矩阵 Carray[i] 的 leading dimension，ldc >= max(1, n)，Host 内存 |
| infoArray | 输出 | int* | 大小为 batchSize 的数组，infoArray[i] = 0 表示求逆成功；= k > 0 表示 U(k,k) == 0（求逆失败），Device 内存 |
| batchSize | 输入 | int | 指针数组中包含的矩阵数量，batchSize >= 0，Host 内存 |

#### 约束说明

- n >= 0, batchSize >= 0
- lda >= max(1, n), ldc >= max(1, n)
- n == 0 或 batchSize == 0 时直接返回成功，不启动 Kernel
- Carray[i] 的内存空间不可与 Aarray[i] 重叠
- 调用前必须先使用 aclblasSgetrfBatched 完成 LU 分解

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

    constexpr int n = 2;
    constexpr int lda = n;
    constexpr int ldc = n;
    constexpr int batchSize = 1;

    float hA[lda * n] = {1.0f, 0.0f, 0.0f, 1.0f};
    float hC[ldc * n] = {0.0f};
    int hPivot[n] = {0, 1};
    int hInfo[batchSize] = {0};

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    void *dA = nullptr, *dC = nullptr, *dPivot = nullptr, *dInfo = nullptr;
    void *dAPtrs = nullptr, *dCPtrs = nullptr;
    aclrtMalloc(&dA, sizeof(hA), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC, sizeof(hC), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dPivot, sizeof(hPivot), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dInfo, sizeof(hInfo), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dAPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, sizeof(hA), hA, sizeof(hA), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dC, sizeof(hC), hC, sizeof(hC), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dPivot, sizeof(hPivot), hPivot, sizeof(hPivot), ACL_MEMCPY_HOST_TO_DEVICE);

    float *dAPtrHost = static_cast<float*>(dA);
    float *dCPtrHost = static_cast<float*>(dC);
    aclrtMemcpy(dAPtrs, sizeof(float*), &dAPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dCPtrs, sizeof(float*), &dCPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasSgetriBatched(
        handle, n,
        reinterpret_cast<const float* const*>(dAPtrs), lda,
        static_cast<const int*>(dPivot),
        reinterpret_cast<float* const*>(dCPtrs), ldc,
        static_cast<int*>(dInfo), batchSize);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hC, sizeof(hC), dC, sizeof(hC), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hInfo, sizeof(hInfo), dInfo, sizeof(hInfo), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("info[0] = %d\n", hInfo[0]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            printf("C[%d][%d] = %f\n", i, j, hC[j * ldc + i]);

    aclrtFree(dA);
    aclrtFree(dC);
    aclrtFree(dPivot);
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
