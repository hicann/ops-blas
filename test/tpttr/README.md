# tpttr 算子

## 功能描述

三角压缩格式转常规格式：将存储于压缩格式（packed format）中的三角矩阵展开为常规二维矩阵。

- `uplo == ACLBLAS_LOWER`：将 `AP` 的元素复制到 `A` 的下三角，上三角保持原值不变
- `uplo == ACLBLAS_UPPER`：将 `AP` 的元素复制到 `A` 的上三角，下三角保持原值不变

压缩格式 `AP` 按列优先存储，长度为 `n*(n+1)/2`。

## 接口定义

```cpp
aclblasStatus_t aclblastpttr(aclblasHandle_t handle,
                              aclblasFillMode_t uplo,
                              int n,
                              const float *AP,
                              float *A,
                              int lda);
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle_t | ACL 流句柄 |
| uplo | aclblasFillMode_t | `ACLBLAS_LOWER`(122) 或 `ACLBLAS_UPPER`(121) |
| n | int | 方阵维数 |
| AP | const float* | 压缩格式存储的三角矩阵（device），长度 n*(n+1)/2 |
| A | float* | 输出常规矩阵（device），维度 lda x n |
| lda | int | A 的 leading dimension，lda >= max(1, n) |

## 数学公式

LOWER 模式：对每列 j，i = j .. n-1：`A[j*lda + i] = AP[colOffset(j) + (i-j)]`

UPPER 模式：对每列 j，i = 0 .. j：`A[j*lda + i] = AP[colOffset(j) + i]`

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
    std::vector<float> AP(n*(n+1)/2);  // packed triangular
    std::vector<float> A(lda * n);
    // ... fill AP ...

    aclblasHandle_t handle;
    aclblasCreate(&handle);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    aclblastpttr(handle, ACLBLAS_LOWER, n, AP.data(), A.data(), lda);
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
- 按列多核切分，DataCopyPad 双缓冲流水线
- SCALAR-bottleneck (~82%)，逐列同步为固有开销
