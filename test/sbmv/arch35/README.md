# SBMV Test

## 描述

本测试用于验证 `aclblasSsbmv`（Symmetric Band Matrix-Vector Multiplication）算子的正确性。

## 测试内容

### 阶段 1: 参数异常校验（10 个用例）

- 无效的 uplo 模式
- k < 0
- lda < k + 1
- incx == 0 / incy == 0
- alpha/beta/A/x/y 为 nullptr

### 阶段 2: L0 门槛级功能用例（5 个用例）

| 编号 | 描述 | n | k | uplo | lda |
|------|------|---|---|------|-----|
| TC-L0-01 | 正常-上三角 | 32 | 2 | UPPER | 3 |
| TC-L0-02 | 正常-下三角 | 32 | 3 | LOWER | 4 |
| TC-L0-03 | 边界-空矩阵 | 0 | 0 | LOWER | 1 |
| TC-L0-04 | 边界-纯对角 | 32 | 0 | LOWER | 1 |
| TC-L0-05 | 边界-单元素 | 1 | 0 | LOWER | 1 |

L0 用例使用 incx=1, incy=1, alpha=0.8, beta=1.2。

### 阶段 3: L1 步长测试用例（12 个用例）

覆盖 incx/incy 为 -1, -2, 2, 3 等正负步长组合，UPPER/LOWER 各一。

### 阶段 4: GEN 泛化用例（6 个用例）

覆盖 n 从 0 到 4096 不同数量级的典型场景。

## 编译和运行

```bash
# 源环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=sbmv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=sbmv --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/sbmv/sbmv_test
```

## 精度标准

- 开发调试期: atol=1e-3, rtol=1e-4
- 正式验收期: MERE < 2^-13, MARE < 10 * 2^-13