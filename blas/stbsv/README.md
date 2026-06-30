# Stbsv算子

## 算子概述

Stbsv 算子实现了三角带状方程组求解操作，核心运算为：`op(A) * x = b`，其中 A 为 n×n 三角带状矩阵（带宽 k），结果原地覆盖到输入向量 x 中。

数学表达式：

```text
op(A) * x = b    (x 原地更新为解)
```

其中 `op(A)` 由 trans 参数决定：`A`（N）、`A^T`（T）或 `A^H`（C）。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStbsv | 单精度三角带状方程组求解 |

## 算子执行接口

### aclblasStbsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbsv(aclblasHandle_t handle,
                              aclblasFillMode_t uplo,
                              aclblasOperation_t trans,
                              aclblasDiagType_t diag,
                              int n, int k,
                              const float *A, int lda,
                              float *x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定三角区域：ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(111)、ACLBLAS_OP_T(112) 或 ACLBLAS_OP_C(113)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(131) 或 ACLBLAS_UNIT(132)，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| k | 输入 | int | 带宽（super/sub-diagonal 数量），k >= 0，Host 内存 |
| A | 输入 | const float\*（FP32） | 带状矩阵，列主序存储，维度 (k+1)×n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda > k，Host 内存 |
| x | 输入/输出 | float\*（FP32） | 解向量，长度至少 1 + (n-1)\*\|incx\|，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0 且 incx != INT_MIN，Host 内存 |

#### 约束说明

- n >= 0，参数校验通过后 n == 0 时直接返回成功
- k >= 0
- lda > k
- incx != 0 且 incx != INT_MIN
- k == 0 且 diag == UNIT 时直接返回成功（x 不变）
- 算子输入 shape 为 [(k+1)×n]、[n]，输出 shape 为 [n]
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    int n = 4, k = 1, lda = 2, incx = 1;

    // 下三角单位对角带状矩阵 A (lda x n, 列主序)，解 A x = b
    float aHost[lda * n] = {1.0f, 1.0f, 0.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 1.0f};
    float xHost[n] = {1.0f, 1.0f, 1.0f, 1.0f};

    size_t aBytes = lda * n * sizeof(float);
    size_t xBytes = n * sizeof(float);
    void *aDev = nullptr, *xDev = nullptr;
    aclrtMalloc(&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDev, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStbsv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                 n, k, static_cast<const float *>(aDev), lda,
                 static_cast<float *>(xDev), incx);

    aclrtSynchronizeStream(stream);

    aclrtFree(aDev);
    aclrtFree(xDev);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
