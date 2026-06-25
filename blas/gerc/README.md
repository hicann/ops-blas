# Gerc算子

## 算子概述

Gerc（Complex Rank-1 Update with Conjugate）算子实现了复数矩阵的共轭秩-1更新操作。Gerc 目前仅有 arch22 实现，无 arch35 实现。

数学表达式：

```
A = alpha * x * conj(y^T) + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCgerc | 复数矩阵共轭秩-1更新 |

## 算子执行接口

### aclblasCgerc

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasCgerc(const int64_t m, const int64_t n, const void *alpha, const void *x, const int64_t incx, const void *y, const int64_t incy, void *A, const int64_t lda, void *stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| m | 输入 | int64_t | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int64_t | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const void*（FP32 complex） | 复数标量，存储为 [real, imag]，Host 内存 |
| x | 输入 | const void*（FP32 complex） | 复数向量 x，长度为 m，Device 内存 |
| incx | 输入 | int64_t | 向量 x 的步长，Host 内存 |
| y | 输入 | const void*（FP32 complex） | 复数向量 y，长度为 n，Device 内存 |
| incy | 输入 | int64_t | 向量 y 的步长，Host 内存 |
| A | 输入/输出 | void*（FP32 complex） | 复数矩阵 A，大小为 m×n，原地更新，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维度，Host 内存 |
| stream | 输入 | void* | ACL 流，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- incx != 0, incy != 0
- lda >= max(1, m)
- 输入向量 x 和 y 必须是有效的复数数组
- 矩阵 A 必须有足够的空间存储 m×n 个复数
- 步长参数 incx 和 incy 当前仅支持值为 1
- 主维度 lda 当前应等于 n

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <complex>

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);

    int64_t m = 4, n = 4;
    int64_t incx = 1;
    int64_t incy = 1;
    int64_t lda = n;
    std::complex<float> alpha(1.0f, 0.0f);

    uint8_t *dx, *dy, *dA;
    aclrtMalloc((void **)&dx, 2 * m * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dy, 2 * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dA, 2 * m * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclblasCgerc(handle, m, n, alpha, dx, incx, dy, incy, dA, lda);

    aclrtStreamSynchronize(nullptr);

    aclrtFree(dx);
    aclrtFree(dy);
    aclrtFree(dA);
    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
