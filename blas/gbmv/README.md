# Gbmv算子

## 算子概述

BLAS Gbmv（General Banded Matrix-Vector Multiplication）算子实现了带状矩阵与向量的乘法运算，针对带状矩阵的稀疏存储特性进行了优化，支持转置操作和多核并行归约。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgbmv | 单精度浮点带状矩阵-向量乘法 |

## 算子执行接口

### aclblasSgbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgbmv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数下同 T），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| kl | 输入 | int | A 的下带宽（主对角线以下的非零对角线数），Host 内存 |
| ku | 输入 | int | A 的上带宽（主对角线以上的非零对角线数），Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host/Device 内存 |
| A | 输入 | const float*（FP32） | 带状矩阵，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 带状矩阵 A 存储的主维长度，lda >= kl + ku + 1，Host 内存 |
| x | 输入 | const float*（FP32） | 向量，trans='N' 时包含 n 个元素，否则包含 m 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数。如果 beta == 0，则 y 不必是有效输入，Host/Device 内存 |
| y | 输入/输出 | float*（FP32） | 向量，trans='N' 时包含 m 个元素，否则包含 n 个元素，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，Host 内存 |

#### 约束说明

- lda >= kl + ku + 1
- incx != 0, incy != 0

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

    constexpr int m = 3;
    constexpr int n = 3;
    constexpr int kl = 1;
    constexpr int ku = 1;
    constexpr int lda = kl + ku + 1;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = m * sizeof(float);
    constexpr size_t aSize = lda * n * sizeof(float);

    float alpha = 1.0f;
    float beta = 0.0f;
    float hX[n] = {1.0f, 1.0f, 1.0f};
    float hY[m] = {0.0f};
    float hA[lda * n] = {0.0f, 1.0f, 3.0f,
                         2.0f, 1.0f, 3.0f,
                         2.0f, 1.0f, 0.0f};

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    float *dA = nullptr, *dX = nullptr, *dY = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dA), aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dX), xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dY), ySize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dY, ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasSgbmv(
        handle, ACLBLAS_OP_N, m, n, kl, ku, &alpha,
        dA, lda, dX, incx, &beta, dY, incy);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hY, ySize, dY, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < m; i++)
        printf("hY[%d] = %f\n", i, hY[i]);

    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
