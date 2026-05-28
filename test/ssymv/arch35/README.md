# SYMV Test

## 描述

本测试用于验证 `aclblasSsymv`（Symmetric Matrix-Vector Multiplication）算子的正确性。

## 测试内容

### 阶段 1: L0 门槛级功能用例（6 个用例）

| 编号 | 描述 | n | uplo | lda | incx | incy | alpha | beta |
|------|------|---|------|-----|------|------|-------|------|
| L001 | 正常-上三角 | 128 | UPPER | 128 | 1 | 1 | 2.0 | 0.0 |
| L002 | 正常-下三角 | 128 | LOWER | 128 | 1 | 1 | 2.0 | 0.0 |
| L003 | alpha=0, beta=1 | 64 | UPPER | 64 | 1 | 1 | 0.0 | 1.0 |
| L004 | 边界-单元素 | 1 | UPPER | 1 | 1 | 1 | 2.0 | 0.0 |
| L005 | 非零 beta | 64 | BOTH | 64 | 1 | 1 | -1.5 | 0.5 |
| L006 | 边界-空矩阵 | 0 | UPPER | 1 | 1 | 1 | 1.0 | 0.0 |

### 阶段 2: L1 参数组合用例（8 个用例）

| 编号 | 参数特点 |
|------|---------|
| L101 | incx=2, UPPER, n=64, 稀疏 x 读取 |
| L102 | incx=-1, UPPER, n=64, 负 stride |
| L103 | incx=-2, LOWER, n=64, 负 stride |
| L104 | incy=2, UPPER, n=64, 稀疏 y 写入 |
| L105 | incy=-1, LOWER, n=64, 负 stride |
| L106 | lda=n+32, UPPER, n=64, lda > n |
| L107 | n=512, UPPER+LOWER, 中规模 |
| L108 | n=4096, UPPER+LOWER, 大规模 |

### 阶段 3: L2 异常入参校验（6 个用例）

| 编号 | 异常条件 | 期望返回值 |
|------|---------|-----------|
| L201 | n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| L202 | incx == 0 | ACLBLAS_STATUS_INVALID_VALUE |
| L203 | incy == 0 | ACLBLAS_STATUS_INVALID_VALUE |
| L204 | lda < max(1, n) | ACLBLAS_STATUS_INVALID_VALUE |
| L205 | handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| L206 | alpha == nullptr | ACLBLAS_STATUS_INVALID_VALUE |

## 编译和运行

```bash
# 源环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=ssymv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=ssymv --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/ssymv/ssymv_test
```

## 精度标准

- 正式验收: MERE < 2^-13 ≈ 1.22e-4, MARE < 10·MERE
