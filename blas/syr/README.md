# Syr算子

## 算子概述

syr (Symmetric Rank-1 Update) 实现对称矩阵的秩-1更新操作。该算子将 `alpha * x * x^T` 加到对称矩阵 A 的指定三角区域，仅上三角或下三角区域被引用和更新。

数学表达式：

```
A := alpha * x * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyr | 单精度对称秩-1更新 |

## 算子执行接口

### aclblasSsyr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyr(aclblasHandle_t handle, aclblasFillMode uplo, const int n, const float* alpha, const float* x, const int incx, float* A, const int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数指针，Host/Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量指针，长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0，Host 内存 |
| A | 输入/输出 | float*（FP32） | 对称矩阵指针，维度 (lda, n)，Device 内存 |
| lda | 输入 | int | A 的主维度，lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- lda >= max(1, n)

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int lda = n;
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t aSize = n * n * sizeof(float);

    float hX[n] = {1.0f, 2.0f};
    float hA[n * n] = {0.0f};
    float alpha = 1.0f;

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    float *dX = nullptr;
    float *dA = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dX), xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dA), aSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dX, xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dA, aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasSsyr(
        handle, ACLBLAS_LOWER, n, &alpha,
        dX, incx, dA, lda);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hA, aSize, dA, aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            printf("hA[%d][%d] = %f\n", i, j, hA[j * lda + i]);

    aclrtFree(dX);
    aclrtFree(dA);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
