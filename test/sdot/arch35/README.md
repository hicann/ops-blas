# SDOT Test

## 描述

本测试用于验证 `aclblasSdot`（Single-precision Vector Dot Product）算子的正确性。

## 测试内容

### 阶段 1: L0 门槛级功能用例（12 个用例）

| 编号 | 描述 | n | incx | incy |
|------|------|---|------|------|
| TC_L0_001 | 基础点积-小规模 | 8 | 1 | 1 |
| TC_L0_002 | 中等规模 | 1024 | 1 | 1 |
| TC_L0_003 | 大规模 | 8192 | 1 | 1 |
| TC_L0_004 | 交替符号 | 1024 | 1 | 1 |
| TC_L0_005 | 随机值 | 2048 | 1 | 1 |
| TC_L0_006 | 全零向量 | 1024 | 1 | 1 |
| TC_L0_007 | 零长度 | 0 | 1 | 1 |
| TC_L0_008 | 单元素 | 1 | 1 | 1 |
| TC_L0_009 | 非单位步长 incx=2 | 512 | 2 | 1 |
| TC_L0_010 | 非单位步长 incy=3 | 512 | 1 | 3 |
| TC_L0_011 | 负步长 incx=-1 | 256 | -1 | 1 |
| TC_L0_012 | 负步长 incy=-1 | 256 | 1 | -1 |

## 编译和运行

```bash
# 源环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=sdot --soc=ascend950

# 编译并运行测试
bash build.sh --ops=sdot --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/sdot/sdot_test
```

## 精度标准

- 开发调试期: atol=1e-3
- 正式验收期: atol=1e-5
