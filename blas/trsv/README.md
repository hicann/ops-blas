# Trsv算子

## 算子概述

trsv (Triangular matrix Solve) 求解三角线性系统。该算子支持前向和后向求解，支持单位对角线和非单位对角线，支持转置和共轭转置。

数学表达式：

```
op(A) * x = b
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrsv | 单精度三角矩阵求解 |

## 算子执行接口

### aclblasStrsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrsv(aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n, const float *A, int64_t lda, float *x, int64_t incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | ACLBLAS_UPPER(121) — A 为上三角矩阵；ACLBLAS_LOWER(122) — A 为下三角矩阵，Host 内存 |
| trans | 输入 | aclblasOperation | ACLBLAS_OP_N(111) — op(A) = A；ACLBLAS_OP_T(112) — op(A) = A^T；ACLBLAS_OP_C(113) — op(A) = A^H（FP32 下与 T 等价），Host 内存 |
| diag | 输入 | aclblasDiagType | ACLBLAS_NON_UNIT(131) — 对角元从 A 读取；ACLBLAS_UNIT(132) — 对角元固定为 1，Host 内存 |
| n | 输入 | int64_t | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功，Host 内存 |
| A | 输入 | const float*（FP32） | n x lda 三角矩阵指针，仅相关三角部分被访问，Device 内存 |
| lda | 输入 | int64_t | A 的 leading dimension，lda >= max(1, n)，Host 内存 |
| x | 输入/输出 | float*（FP32） | 输入时存储右端向量 b，输出时原地覆盖为解向量 x，Device 内存 |
| incx | 输入 | int64_t | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储，Host 内存 |

#### 约束说明

- n >= 0，n == 0 时为空操作直接返回成功
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- lda >= max(1, n)
- incx != 0（可正可负）
- A、x 不可为 nullptr

#### 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);

    constexpr int64_t n = 4;
    constexpr int64_t incx = 1;
    constexpr int64_t lda = 4;
    constexpr size_t aSize = n * lda * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    float hA[n * lda] = {
        1.0f, 2.0f, 4.0f, 7.0f,
        0.0f, 3.0f, 5.0f, 8.0f,
        0.0f, 0.0f, 6.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 10.0f
    };
    float hX[n] = {1.0f, 4.0f, 9.0f, 16.0f};

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    float *dA = nullptr;
    float *dX = nullptr;
    aclrtMalloc((void**)&dA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasStrsv(
        handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, dA, lda, dX, incx);

    aclrtStreamSynchronize(nullptr);

    aclrtMemcpy(hX, xSize, dX, xSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dA);
    aclrtFree(dX);
    aclrtDestroyStream(stream);

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
