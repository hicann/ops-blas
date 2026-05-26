# SPMV Test

## 描述

本测试用于验证 `aclblasSspmv`（Symmetric Packed Matrix-Vector Multiplication）算子在 ascend950(arch35) 上的正确性。

## 测试内容

### 阶段 1: 参数异常校验（10 个用例）

- 无效的 uplo 模式
- n < 0
- incx == 0 / incy == 0
- alpha/beta/AP/x/y 为 nullptr
- handle 为 nullptr

### 阶段 2: L0 门槛级功能用例（5 个用例）

| 编号 | 描述 | n | uplo | incx | incy | alpha | beta |
|------|------|---|------|------|------|-------|------|
| TC-L0-01 | 正常-上三角 | 4 | UPPER | 1 | 1 | 0.8 | 1.2 |
| TC-L0-02 | 正常-下三角 | 4 | LOWER | 1 | 1 | 0.8 | 1.2 |
| TC-L0-03 | 边界-空矩阵 | 0 | LOWER | 1 | 1 | 1.0 | 0.0 |
| TC-L0-04 | 边界-单元素 | 1 | LOWER | 1 | 1 | 0.8 | 1.2 |
| TC-L0-05 | 正常-中等规模 | 128 | LOWER | 1 | 1 | 0.8 | 1.2 |

### 阶段 3: GEN 泛化用例（5 个用例）

覆盖 n 从 512 到 4096 的随机规模，UPPER/LOWER 混合，正负步长和不同 alpha/beta 组合。

## 编译和运行

```bash
# 源环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=spmv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=spmv --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/spmv/spmv_test
```

## 精度标准

- 开发调试期: atol=1e-3, rtol=1e-4
- 正式验收期: MERE < 2^-13, MARE < 10 * 2^-13
