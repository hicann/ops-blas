# stpttr 算子

## 功能描述

三角压缩格式转常规格式：将存储于压缩格式（packed format）中的三角矩阵展开为常规二维矩阵。

- `uplo == ACLBLAS_LOWER`：将 `AP` 的元素复制到 `A` 的下三角，上三角保持原值不变
- `uplo == ACLBLAS_UPPER`：将 `AP` 的元素复制到 `A` 的上三角，下三角保持原值不变

## 目录结构

```
blas/tpttr/stpttr/arch35/     # 算子实现（tpttr=操作族, stpttr=float32, arch35=ascend950）
test/stpttr/arch35/            # 测试代码
```

## 接口定义

```cpp
aclblasStatus_t aclblasStpttr(aclblasHandle_t handle, aclblasFillMode_t uplo,
                               int n, const float *AP, float *A, int lda);
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle_t | ACL 流句柄 |
| uplo | aclblasFillMode_t | `ACLBLAS_LOWER`(122) 或 `ACLBLAS_UPPER`(121) |
| n | int | 方阵维数 |
| AP | const float* | 压缩格式三角矩阵（device），长度 n*(n+1)/2 |
| A | float* | 输出常规矩阵（device），维度 lda x n |
| lda | int | A 的 leading dimension，lda >= max(1, n) |

## 编译运行

```bash
bash build.sh --ops=stpttr --run --soc=ascend950
```
