# aclblasCgemmBatched 真机测试

本目录包含 Ascend 950PR 上的任务 CSV 精度测试、边界回归和固定性能契约。CSV 共 1,200
个用例：前 1,000 个功能/精度用例覆盖基础尺寸、尺寸扫描、标量、batch、leading
dimension、输入分布、边界负向和扩展组合；后 200 个 `TC_PF` 行保留任务性能输入。
GTest 性能契约单独覆盖任务书指定的三个验收形状。

| 精度类别 | 数量 |
|---|---:|
| `TC_L0` 基础转置组合 | 18 |
| `TC_SQ` 尺寸扫描 | 23 |
| `TC_AB` alpha/beta | 72 |
| `TC_BC` batch 扫描 | 13 |
| `TC_LD` leading dimension/padding | 12 |
| `TC_FL` 输入分布与 Inf/NaN | 6 |
| `TC_CV` 中等形状覆盖 | 72 |
| `TC_ED` 边界与负向 | 29 |
| `TC_EX` 扩展组合 | 755 |
| 合计 | 1,000 |

## Release 构建

在 CANN 9.1 环境中设置工具链变量后执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cgemm_batched_test -j4
```

测试程序位于：

```text
build/test/gemm_batched/cgemm_batched/cgemm_batched_test
```

## 正确性与复测

```bash
# 原包前 1,000 条功能/精度用例，以及两个补充边界回归
./build/test/gemm_batched/cgemm_batched/cgemm_batched_test --gtest_filter='-*TC_PF*'

# 原包后 200 条 TC_PF 输入的逐元素正确性
./build/test/gemm_batched/cgemm_batched/cgemm_batched_test \
  --gtest_filter='CgemmBatched/CgemmBatchedArch35Test.CsvDriven/TC_PF_*'
```

两条命令合计覆盖原包全部 1,200 个 CSV 输入；第一条另覆盖 null handle 和 `k == 0`
非有限 `alpha` 回归。验证将一次 batched 调用的全部逻辑输出聚合，实部和虚部分别应用
99% 混合容差；`ldc` padding 不参与比较。`TC_PF` 的 GTest 用时包含输入生成、CPU golden
和逐元素比对，只用于正确性，不作为 NPU kernel 性能证据。

## NPU 性能

```bash
./build/test/gemm_batched/cgemm_batched/cgemm_batched_test \
  --gtest_filter='CgemmBatchedArch35Test.TC_PF_PerformanceContract'
```

每个形状预热 5 次、计时 60 次，设备流同步后输出 `average_us`。随任务提供的
`gpu_baseline.csv` 定义性能倍率为 `H100 time / NPU time`，要求倍率不低于 `0.4`，
因此验收上限按 `H100 time / 0.4` 计算：

| 用例 | H100 基线 | NPU 验收上限 |
|---|---:|---:|
| 256x256x256, batch=32, NN | 91.439 us | 228.598 us |
| 512x512x512, batch=16, NN | 337.538 us | 843.845 us |
| 1024x1024x1024, batch=8, NN | 1309.025 us | 3272.563 us |

性能用例在超过目标时会失败，这是预期的契约行为；正确性准入过滤该用例，性能结果由
`research/performance-gate.sh` 独立记录。CPU、模拟器或 mock 结果不能作为性能证据。
