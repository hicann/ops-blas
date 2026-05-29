# strttp 算子

## 功能描述

三角常规格式转压缩格式：将常规二维三角矩阵压缩为 packed format 存储。

- `uplo == ACLBLAS_LOWER`：将 `A` 的下三角复制到 `AP`
- `uplo == ACLBLAS_UPPER`：将 `A` 的上三角复制到 `AP`

## 目录结构

```
blas/trttp/strttp/arch35/    # 算子实现（trttp=操作族, strttp=float32, arch35=ascend950）
test/strttp/arch35/           # 测试代码
```

## 接口定义

```cpp
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo,
                               int n, const float *A, int lda, float *AP);
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

## 编译运行

```bash
bash build.sh --ops=strttp --run --soc=ascend950
```

## 测试说明

ST 采用 GTest 参数化 + `strttp_test.csv`，`BlasTest<StrttpParam>` fixture，精度模式为 **EXACT**。

**注意**：`makeBlasArray` 的 size 参数为 `int64_t`，调用时需显式转换：`makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a)`，确保负值 n 正确返回空数组。
