# strmv算子测试

## 算子描述

strmv实现实三角矩阵向量乘法：x = A * x 或 x = A^T * x

其中A为n阶三角矩阵（上三角或下三角），x为n维向量。

## 测试用例

- uplo=U, trans=N, diag=N：上三角矩阵，不转置，非单位对角线
- uplo=L, trans=N, diag=N：下三角矩阵，不转置，非单位对角线
- uplo=U, trans=T, diag=N：上三角矩阵，转置，非单位对角线
- uplo=L, trans=T, diag=N：下三角矩阵，转置，非单位对角线

## 编译运行

```bash
source /opt/nzh/0408/ascend-toolkit/set_env.sh
bash build.sh --ops=strmv --run
```

## 参数说明

| 参数 | 说明 |
|------|------|
| uplo | 上三角(U)或下三角(L) |
| trans | 是否转置(N或T) |
| diag | 对角线是否为单位矩阵(N或U) |
| n | 矩阵阶数 |
| lda | 矩阵行间隔 |
| incx | 向量步长 |