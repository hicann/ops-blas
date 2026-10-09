# aclblasCtbsv 测试说明

> 适配硬件：Ascend 950PR（arch35）  
> CANN：9.1.0  
> 接口：`aclblasCtbsv`  
> golden：cblas（Netlib `cblas_ctbsv`）

本目录是 ops-blas 仓内 CSV + GTest 测试工程，覆盖任务配套的全部自测用例。验收人按下面步骤即可复现精度与性能结果。

## 目录结构

```text
test/tbsv/ctbsv/
├── CMakeLists.txt
├── README.md                 # 本文件
├── ctbsv_param.h             # CSV 参数解析
├── ctbsv_golden.h            # cblas_ctbsv golden
└── arch35/
    ├── ctbsv_npu_wrapper.h   # NPU 调用封装
    ├── ctbsv_test.cpp        # 精度 / 负向 GTest
    ├── ctbsv_test.csv        # 1200 条用例（1000 精度 + 200 性能）
    └── ctbsv_bench.cpp       # 任务书 §3.3 四个典型 case 的独立计时
```

## 用例清单

CSV：`arch35/ctbsv_test.csv`，共 **1200** 条。

### 精度测试 case（1000 条，排除 `TC_PF_`）

| 前缀 | 条数 | 场景 |
|------|------|------|
| TC_L0 | 12 | 小 shape 基础：n=8,k=2，uplo×trans×diag 全组合 |
| TC_SQ | 23 | 尺寸扫描 1→2048（含奇数 / 边界 / 非对齐） |
| TC_BD | 22 | 带宽 0/1/小值/半带/满带 |
| TC_DG | 12 | UNIT 对角不读 vs NON_UNIT |
| TC_LD | 6 | lda = k+1+padding |
| TC_INC | 12 | incx ∈ {±1,±2,±3} |
| TC_FL | 11 | 填充 / Inf/NaN（UNIT 对角不读） |
| TC_CV | 24 | 中等尺寸全枚举 |
| TC_ED | 13 | 零维、空指针、非法枚举(INVALID_ENUM)、非法维度 / 步长(INVALID_VALUE) |
| TC_EX | 865 | 扩展精度采样 |

### 性能测试 case（200 条，`TC_PF_`）

CSV 中 `TC_PF_1001`～`TC_PF_1200` 为性能/内存用例。其中前 4 条与任务书 §3.3 完全一致，由独立二进制 `ctbsv_bench` 用 `aclrtEvent` 计时（warmup 20 + 采样 100）：

| Case | n | k | uplo | trans | diag | lda | incx | 标杆 Avg (us) |
|------|---|---|------|-------|------|-----|------|---------------|
| TC_PF_1001 | 512 | 8 | UPPER | N | NON_UNIT | 9 | 1 | 917.5 |
| TC_PF_1002 | 1024 | 16 | LOWER | N | NON_UNIT | 17 | 1 | 1831.11 |
| TC_PF_1003 | 2048 | 16 | UPPER | T | NON_UNIT | 17 | 1 | 3753.09 |
| TC_PF_1004 | 4096 | 32 | LOWER | C | UNIT | 33 | 1 | 4898.68 |

精度判定：工程侧实部/虚部分别走仓库 `verifyMereMareComplexFloat`（mere=2⁻¹³，mare×10）。双方均溢出（Inf/NaN/超大有限值）的分量按 special-value 对齐；大带宽病态回代若 MERE/MARE 仍超阈值，再用 `op(A)·x ≈ b` 残差兜底。本地复测精度 **1000/1000 通过**。

## 编译

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /path/to/ops-blas
bash build.sh --ops=ctbsv --soc=ascend950
```

产物：

- `build/test/tbsv/ctbsv/ctbsv_test`
- `build/test/tbsv/ctbsv/ctbsv_bench`（若已按 CMake 注册）

## 精度复现

```bash
export LD_LIBRARY_PATH=$PWD/build:/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64:$LD_LIBRARY_PATH

# 全量精度 1000 条（排除性能 TC_PF_）
./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter='-*TC_PF*'

# 核心套件 135 条
./build/test/tbsv/ctbsv/ctbsv_test \
  --gtest_filter='*TC_L0*:*TC_SQ*:*TC_BD*:*TC_DG*:*TC_LD*:*TC_INC*:*TC_FL*:*TC_CV*:*TC_ED*'

# 扩展精度 865 条
./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter='*TC_EX*'

# 也可使用任务包脚本
python3 /path/to/test_cases/verify_accuracy.py --repo $PWD --soc ascend950 --skip-build
```

## 性能复现

```bash
# 推荐：任务书 4 个典型 case，warmup 20 + 采样 100，单位 us
./build/test/tbsv/ctbsv/ctbsv_bench

# 或直接编译独立 bench
g++ -O2 -std=c++17 test/tbsv/ctbsv/arch35/ctbsv_bench.cpp \
  -Iinclude -I$ASCEND_HOME_PATH/include \
  -Lbuild -L$ASCEND_HOME_PATH/x86_64-linux/lib64 \
  -lops_blas -lascendcl -o /tmp/ctbsv_bench
/tmp/ctbsv_bench
```

判定：平均单次耗时 ≤ 任务书标杆即 PASS。

## 内存

不单独申请 workspace。单用例 Device 占用约为：

```text
A: lda * n * sizeof(aclblasComplex)
x: (1 + (n - 1) * |incx|) * sizeof(aclblasComplex)
```

最大典型 case（n=4096, k=32, lda=33, incx=1）：

```text
A = 33 * 4096 * 8 = 1,081,344 B
x = 4096 * 8      = 32,768 B
合计 ≈ 1.06 MiB
```

Host 侧单用例设计预算 ≤ 512 MiB。
