# Trmm算子

## 算子概述

Trmm（Single-precision Triangular Matrix Multiplication）算子实现了单精度浮点三角矩阵与普通矩阵的乘法运算，结果乘以标量 alpha 后写入输出矩阵 C。

数学表达式：

```
LEFT 模式：C := alpha * op(A) * B
RIGHT 模式：C := alpha * B * op(A)
```

其中 A 为三角矩阵（仅存储上三角或下三角），B 为输入矩阵，C 为输出矩阵，op(A) 为 A、A^T 或 A^H。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrmm | 单精度浮点三角矩阵乘法 |

## 算子执行接口

### aclblasStrmm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 strmm 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasStrmm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_UPPER（上三角）或 ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，FP32 等同 T），Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_UNIT（单位三角，对角线为 1）或 ACLBLAS_NON_UNIT（非单位三角），Host 内存 |
| m | 输入 | int | 矩阵 B/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 B/C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 三角矩阵，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，side=LEFT 时 lda >= m，side=RIGHT 时 lda >= n，Host 内存 |
| B | 输入 | const float*（FP32） | m×n 输入矩阵，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维，ldb >= n，Host 内存 |
| C | 输出 | float*（FP32） | m×n 输出矩阵，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维，ldc >= n，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- side=LEFT 时：lda >= m
- side=RIGHT 时：lda >= n
- ldb >= n
- ldc >= n
- alpha 不可为 nullptr
- A、B、C 不可为 nullptr

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr int ldc = 4;
    constexpr size_t aSize = m * lda * sizeof(float);
    constexpr size_t bSize = m * ldb * sizeof(float);
    constexpr size_t cSize = m * ldc * sizeof(float);
    float alpha = 1.0f;

    float hA[m * lda] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.0f, 5.0f, 6.0f, 7.0f,
        0.0f, 0.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 10.0f
    };
    float hB[m * ldb] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    float hC[m * ldc] = {0};

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    aclrtMalloc((void**)&dA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB, bSize, hB, bSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasStrmm(
        handle, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        m, n, &alpha, dA, lda, dB, ldb, dC, ldc);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hC, cSize, dC, cSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    aclrtDestroyStream(stream);

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
