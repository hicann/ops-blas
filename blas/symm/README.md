# Symm算子

## 算子概述

Symm（Single-precision Symmetric Matrix Multiplication）算子实现了单精度浮点对称矩阵与普通矩阵的乘法运算。

数学表达式：

```
LEFT 模式：C := alpha * A * B + beta * C
RIGHT 模式：C := alpha * B * A + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymm | 单精度浮点对称矩阵乘法 |

## 算子执行接口

### aclblasSsymm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_LOWER（下三角）或 ACLBLAS_UPPER（上三角），Host 内存 |
| m | 输入 | int | 矩阵 C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 或 Device 内存 |
| A | 输入 | const float*（FP32） | 对称矩阵，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存（详见约束说明） |
| B | 输入 | const float*（FP32） | m×n 普通矩阵，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维，Host 内存（详见约束说明） |
| beta | 输入 | const float*（FP32） | 标量 beta，不可为 nullptr，Host 或 Device 内存 |
| C | 输入/输出 | float*（FP32） | m×n 矩阵，输入旧值，输出新值，beta!=0 时不可为 nullptr，beta==0 时可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维，Host 内存（详见约束说明） |

#### 约束说明

**通用约束：**

- m >= 0, n >= 0
- side=LEFT 时：lda >= max(1, m)
- side=RIGHT 时：lda >= max(1, n)
- alpha、beta 不可为 nullptr
- A、B 不可为 nullptr（m>0 且 n>0 时）
- beta==0 时 C 可为 nullptr（BLAS 标准：beta==0 时 C 不需要是有效输入）

**arch35（Ascend 950PR / 950DT / Atlas A3）约束：**

- 矩阵 A、B、C 均按列主序存储（column-major），元素 (row, col) 存储于 col*ld + row 位置
- ldb >= max(1, m)
- ldc >= max(1, m)

**arch22（Atlas A2）约束：**

- 矩阵 A、B、C 均按行主序存储（row-major），元素 (row, col) 存储于 row*ld + col 位置
- ldb >= n
- ldc >= n

#### 调用示例

```cpp
// arch35（列主序）调用示例：C = alpha * A * B + beta * C
// side=LEFT, uplo=LOWER, m=4, n=4, alpha=1.0, beta=0.0

#include "acl/acl.h"
#include "cann_ops_blas.h"

// 1. 初始化
aclInit(nullptr);
aclblasHandle_t handle;
aclblasCreateHandle(&handle);

// 2. 准备 Host 数据（列主序存储）
int m = 4, n = 4;
int lda = m, ldb = m, ldc = m;
float alpha = 1.0f;
float beta = 0.0f;

// A (4x4 对称矩阵, 下三角, 列主序, lda=4)
std::vector<float> hA = {
    1.0f, 2.0f, 3.0f, 4.0f,   // col 0
    2.0f, 5.0f, 6.0f, 7.0f,   // col 1
    3.0f, 6.0f, 8.0f, 9.0f,   // col 2
    4.0f, 7.0f, 9.0f, 10.0f   // col 3
};
// B (4x4 普通矩阵, 列主序, ldb=4)
std::vector<float> hB = {
    1.0f, 0.0f, 0.0f, 0.0f,   // col 0
    0.0f, 1.0f, 0.0f, 0.0f,   // col 1
    0.0f, 0.0f, 1.0f, 0.0f,   // col 2
    0.0f, 0.0f, 0.0f, 1.0f    // col 3
};
std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);

// 3. 申请 Device 内存并拷贝数据
size_t aBytes = hA.size() * sizeof(float);
size_t bBytes = hB.size() * sizeof(float);
size_t cBytes = hC.size() * sizeof(float);

void* dA = nullptr;
aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);

void* dB = nullptr;
aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dB, bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

void* dC = nullptr;
aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dC, cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);

// 4. 执行计算
aclblasSsymm(handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER,
    m, n, &alpha,
    static_cast<const float*>(dA), lda,
    static_cast<const float*>(dB), ldb,
    &beta,
    static_cast<float*>(dC), ldc);

// 5. 同步并拷回结果
aclrtStream stream;
aclblasGetStream(handle, &stream);
aclrtSynchronizeStream(stream);
aclrtMemcpy(hC.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

// 6. 释放资源
aclrtFree(dA);
aclrtFree(dB);
aclrtFree(dC);
aclblasDestroyHandle(handle);
aclFinalize();
```
