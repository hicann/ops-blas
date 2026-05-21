# trttp 算子

## 功能描述

三角常规格式转压缩格式：将常规二维三角矩阵压缩为 packed format 存储。

- `uplo == ACLBLAS_LOWER`：将 `A` 的下三角复制到 `AP`
- `uplo == ACLBLAS_UPPER`：将 `A` 的上三角复制到 `AP`

压缩格式 `AP` 按列优先存储，长度为 `n*(n+1)/2`。此算子为 `tpttr` 的逆操作。

## 接口定义

```cpp
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle,
                               aclblasFillMode_t uplo,
                               int n,
                               const float *A,
                               int lda,
                               float *AP);
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle_t | ACL 流句柄 |
| uplo | aclblasFillMode_t | `ACLBLAS_LOWER`(122) 或 `ACLBLAS_UPPER`(121) |
| n | int | 方阵维数 |
| A | const float* | 常规三角矩阵（device），维度 lda x n |
| lda | int | A 的 leading dimension，lda >= max(1, n) |
| AP | float* | 输出压缩格式（device），长度 n*(n+1)/2 |

## 数学公式

LOWER 模式：对每列 j，i = j .. n-1：`AP[colOffset(j) + (i-j)] = A[j*lda + i]`

UPPER 模式：对每列 j，i = 0 .. j：`AP[colOffset(j) + i] = A[j*lda + i]`

其中 `colOffset(j) = j*(2*n - j + 1)/2` (LOWER) 或 `j*(j+1)/2` (UPPER)。

## 使用示例

```cpp
#include "cann_ops_blas.h"
#include <vector>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    int n = 4;
    int lda = 4;
    std::vector<float> A(lda * n);      // full triangular matrix
    std::vector<float> AP(n*(n+1)/2);   // packed output
    // ... fill A ...

    aclblasHandle_t handle;
    aclblasCreate(&handle);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    aclblasStrttp(handle, ACLBLAS_LOWER, n, A.data(), lda, AP.data());
    aclrtSynchronizeStream(stream);

    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

## 性能特点

- 纯数据搬运算子，位精确（无精度损失）
- 基于前缀和的按元素多核切分，DataCopyPad 双缓冲流水线
- SCALAR-bottleneck (~82%)，逐列同步为固有开销
