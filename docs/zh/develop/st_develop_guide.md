# 算子 ST 设计方案与代码编写指导

本文档描述 **gbmv**、**stpttr** 已落地的一套 GTest + CSV 精度 ST 框架，并给出新算子接入时的设计与编码规范。  

---

## 1 框架方案设计

### 1.1 核心设计思想

**CSV 定义用例参数 → GTest 参数化读取并驱动 → 算子测试代码内完成数据准备 / API 调用 / Golden 比对**

| 层次 | 职责 | 由谁承担 |
|------|------|----------|
| 用例参数 | 维度、系数、步长、期望返回值、精度阈值等 | `<op>_testcases.csv` |
| 框架公共能力 | CSV 解析、精度验证、数据生成、device 内存、GTest 辅助 | `test/frame/*.h`（6 个头文件） |
| 算子专属逻辑 | API 调用、device 内存、Golden、异常分支 | `<op>_test.cpp` + `<op>_golden.h` |

设计原则：

1. **CSV 只承载「会变」的参数**，方便用表格软件维护、做 L0/L1/L2 分级。
2. **算子差异写在测试代码里**，不追求通用反射式框架；gbmv 与 stpttr 的异常用例分派均如此。
3. **ST 框架默认 device 指针**：测试侧 H2D → API → D2H，不把 host 指针直接传入 kernel 型 API。
4. **ACL 生命周期在 Test Suite 级初始化一次**（`SetUpTestSuite` / `TearDownTestSuite`），禁止每个用例单独 `aclInit/aclFinalize`（会导致设备挂死）。
5. **自定义 `main()`** 接收配置目录路径；`build.sh --run` 传入 `build/test/<op>/`。

### 1.2 CSV 用例表设计

每个算子一个 CSV，命名 `<op_name>_testcases.csv`。首行为列头，每行一条用例，空字段表示该算子不使用该列。

#### gbmv 示例（精度 + 功能）

```csv
case_id,level,description,data_type,trans,m,n,kl,ku,alpha_real,beta_real,incx,incy,lda,seed,expect_success,mere_threshold,mare_multiplier
TC_L0_001,L0,Small square banded matrix (trans=N),fp32,N,8,8,2,2,1.5,0.8,1,1,5,20260516,true,0.0001220703125,10
TC_L1_007,L1,Empty rows m=0 (immediate return),fp32,N,0,8,0,0,1.0,0.0,1,1,1,20260528,false,
```

#### stpttr 示例（参数校验 + 精度）

```csv
case_id,level,description,uplo,n,lda,expect_success
TC_L0_01,L0,handle_null,LOWER,5,5,false
TC_L1_41,L1,large_n_10240,LOWER,10240,10240,true
```

#### 通用列定义

列名与 `test/frame/types.h` 中 `TestCaseConfig` 字段一一对应（snake_case）：

| 列名 | 类型 | 说明 |
|------|------|------|
| case_id | string | 用例编号，作为 GTest 参数名（`-` 会替换为 `_`） |
| level | enum | L0 / L1 / L2 |
| description | string | 用例描述；stpttr 也用于解析特殊值类型 |
| data_type | enum | fp32 / fp16 等（预留） |
| trans | string | N / T / C |
| m, n, k | int64 | 矩阵维度 |
| kl, ku | int64 | 带状矩阵上下带宽 |
| lda, ldb, ldc | int64 | leading dimension |
| alpha_real, beta_real | float | 标量系数 |
| incx, incy | int64 | 向量步长（支持负值） |
| uplo | string | UPPER / LOWER / INVALID 等 |
| seed | uint32 | 随机种子 |
| expect_success | bool | `true` 走精度比对；`false` 只校验返回码 |
| mere_threshold, mare_multiplier | float | MERE_MARE 精度（gbmv 使用） |
| abs_tol, rel_tol | float | ABS / REL / COMBINED 精度（可选） |

**规则**：算子只填相关列；加载后未出现的列保持 `std::nullopt`，代码侧用 `value_or()` 设默认值。

### 1.3 配置加载（仅 CSV）

`ConfigLoader::loadAllForOp(configDir, opName)` 查找：

1. `<opName>_testcases.csv` — **必需**（新框架路径）
2. `<opName>_config.json` — **可选**（若存在则补充默认精度等；gbmv/stpttr 当前未使用）
3. `<opName>_testcases.json` — **回退**（旧单文件 JSON 格式，向后兼容）

```cpp
static std::vector<TestCaseConfig> loadGbmvCases() {
    auto [cases, cfg] = ConfigLoader::loadAllForOp(g_configDir, "gbmv");
    (void)cfg;
    return cases;
}
```

加载 CSV 时，`ConfigLoader` 会根据 CSV 中的精度列构建每条用例的 `tc.verifyCfg`：

- 有 `mere_threshold` → `PrecisionMode::MERE_MARE`
- 有 `abs_tol` / `rel_tol` → 对应 ABS / REL / COMBINED
- 均无 → 使用默认 `ABS`（可在 `load*Cases()` 中按算子硬编码覆盖，见 stpttr）

### 1.4 项目目录结构

```
ops-blas/
  test/
    frame/                          # ST 框架公共头文件（6 个，header-only）
      types.h                    # TestCaseConfig / OpConfig / VerifyConfig
      config.h                   # ConfigLoader：CSV / JSON 解析
      verify.h                   # Verifier + verifyDenseVector / verifyStridedVector
      device.h                   # DeviceBuffer / allocAndCopyToDevice
      data.h                     # DataGenerator 填充原语
      gtest.h                    # GTest 参数名生成
    utils/                          # 旧版 / 非 ST 框架工具
      error_check.h                 # CHECK_ACLRT 等宏（perf/直调测试用）
      golden.h                      # 旧版 matmul 工具（legacy）
    gbmv/
      arch35/
        gbmv_testcases.csv
        gbmv_test.cpp
        gbmv_test_utils.h           # gbmv 专用数据生成与验证
        gbmv_golden.h
      CMakeLists.txt
    stpttr/
      arch35/
        stpttr_testcases.csv
        stpttr_test.cpp
        stpttr_golden.h
      CMakeLists.txt
  docs/zh/develop/
    st_develop_guide.md             # 本文档
```

**CMake 约定**：精度 ST 使用 `ops_blas_add_gtest_tests(${OPS_BLAS} <op>_test)`；构建时自动将 CSV 复制到可执行文件同目录（`cmake/test.cmake` 中 `_ops_blas_copy_test_config_files`）。

---

### 1.5 框架公共头文件说明（`test/frame/`）

#### types.h

- `TestCaseConfig`：单条用例的全部参数 + `VerifyConfig verifyCfg`
- `OpConfig`：可选 JSON 元数据（默认精度等）

#### config.h

- `ConfigLoader::loadAllForOp(dir, opName)`：返回 `{cases, opCfg}`
- 依赖 `types.h` 与 `verify.h`（构建 `verifyCfg`）

#### verify.h

- `PrecisionMode`：ABS / REL / COMBINED / MERE_MARE / EXACT / INTEGER
- `Verifier::verifyVector` / `verifyScalar`：按模式比对并打印 `[case_id] PASSED/FAILED`
- `verifyDenseVector` / `verifyStridedVector`：基于 `TestCaseConfig` 的通用比对封装

#### data.h

| 函数 | 适用场景 |
|------|----------|
| `fillBandedMatrix` | 带状矩阵 A |
| `fillStridedVector` | 带步长的 x / y |
| `fillPackedMatrix` | packed 三角 AP |
| `fillTriangularMatrix` | 稠密三角矩阵 |

#### device.h

| 函数/类 | 用途 |
|---------|------|
| `DeviceBuffer` | RAII device 内存 |
| `allocAndCopyToDevice` | host vector → device buffer 批量拷贝 |
| `adjustStridedBase` | BLAS 负步长基指针调整（验证阶段使用；传 API 前需确认 kernel 约定） |

#### gtest.h

```cpp
INSTANTIATE_TEST_SUITE_P(Gbmv, GbmvTest,
    ::testing::ValuesIn(loadGbmvCases()),
    gtestParamNameFromTestCase);  // 参数名 = case_id（'-' → '_'）
```

#### 算子专用 helper

算子特有的数据生成、kernel 参数解析、验证逻辑放在 `test/<op>/` 下，例如 `gbmv_test_utils.h`，**不要**放入 `test/frame/`。

---

### 1.6 GTest 参数化测试入口

标准模板（所有新算子 ST 遵循）：

```cpp
static std::string g_configDir = ".";

class XxxTest : public ::testing::TestWithParam<TestCaseConfig> {
protected:
    static void SetUpTestSuite() {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclblasCreate(&handle_);
        aclrtCreateStream(&stream_);
        aclblasSetStream(handle_, stream_);
    }
    static void TearDownTestSuite() { /* 逆序释放 */ }
    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

INSTANTIATE_TEST_SUITE_P(Xxx, XxxTest,
    ::testing::ValuesIn(loadXxxCases()),
    gtestParamNameFromTestCase);

TEST_P(XxxTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();
    // Setup → API → Sync → Golden → Verify
}

int main(int argc, char* argv[]) {
    g_configDir = (argc > 1) ? argv[1] : ".";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**注意**：`loadXxxCases()` 在 `INSTANTIATE_TEST_SUITE_P` 展开时调用，此时 `main()` 尚未执行，因此 `g_configDir` 在注册参数化用例前默认为 `"."`。`build.sh --run` 从 `build/test/<op>/` 启动，CSV 已通过 POST_BUILD 复制到该目录，故可正常加载。

---

### 1.7 GTest Fixture 与 ACL 生命周期

```
SetUpTestSuite (一次)
  aclInit → aclrtSetDevice → aclblasCreate → aclrtCreateStream → aclblasSetStream
       ↓
  TEST_P × N  (每条 CSV 一行)
       ↓
TearDownTestSuite (一次)
  aclrtDestroyStream → aclblasDestroy → aclrtResetDevice → aclFinalize
```

- 所有计算 API 统一传入 `aclblasHandle_t handle`（见 `include/cann_ops_blas.h`）。
- 异步算子路径在 API 调用后需 `aclrtSynchronizeStream(stream_)` 再读结果。

---

### 1.8 已验证的算子模式（统一 device 指针）

开发新算子前，先读 `blas/<op>/*_host.cpp` 与 kernel 源码，确认 **数据指针为 device 地址** 及 **负步长索引方式**。

#### 标准 device 流程（gbmv / stpttr / strttp）

```
测试代码                         API host 层                    Kernel
host 填数 → H2D  ──device ptr──►  直接 launch  ──►  *_kernel_do
sync → D2H  ◄── 读回结果
```

**gbmv 要点**：

- `aclblasSgbmv` 的 A/x/y 均为 device 指针；测试用 `allocAndCopyToDevice` 分配并 H2D
- kernel 内部用 `(dim-1-i)*(-inc)` 处理负步长，测试传 **buffer 基址**，勿做 BLAS 指针调整
- 精度：CSV 填 `mere_threshold` + `mare_multiplier`；验证用 `verifyGbmvResult`

**stpttr / strttp 要点**：

- 正常路径：`aclrtMalloc` → H2D → API → sync → D2H
- 精度：在 `load*Cases()` 中设 `tc.verifyCfg.mode = PrecisionMode::EXACT`
- 异常用例：`expect_success=false` 时按 `case_id` 分派具体非法参数组合

---

### 1.9 精度验证策略

| 算子 | 模式 | 配置来源 |
|------|------|----------|
| gbmv | MERE_MARE | CSV 列 `mere_threshold`, `mare_multiplier` |
| stpttr | EXACT | `loadStpttrCases()` 硬编码；golden 用 sentinel (-999) 标记未写入区域 |

`Verifier` 输出示例：

```
[TC_L0_001] MERE=0.00001 MARE=0.00005 (threshold=0.000122, outlier_limit=0.00122)
[TC_L0_001] PASSED (MERE < threshold && MARE < 10*threshold, 0 outliers out of 8 elements)
```

---

### 1.10 构建与运行

```bash
source <cann>/set_env.sh
cd ops-blas
bash build.sh --ops=gbmv,stpttr --soc=ascend950 --run
```

- `build.sh --run` 执行 `build/test/<op>/<op>_test build/test/<op>`，第二个参数即 CSV 所在目录。
- 手动运行：`./build/test/gbmv/gbmv_test build/test/gbmv`
- CMake：`ops_blas_add_gtest_tests` 只链接 `gtest`（**不**链接 `gtest_main`，因使用自定义 main）。

---

### 1.11 框架特点小结

| 优点 | 说明 |
|------|------|
| CSV 易维护 | 增删用例不改 C++ 代码 |
| GTest 集成 | 过滤、并行、CI 友好（`--gtest_filter=Gbmv*/TC_L0_001`） |
| 算子逻辑清晰 | host/device、异常分支在 test 内可读可调试 |
| 公共工具复用 | 解析、验证、数据生成、device 内存集中在 `test/frame/`（6 个头文件） |

| 当前限制 | 说明 |
|----------|------|
| 非全自动框架 | API 调用、内存模型、golden 均在算子 test 中手写 |
| utils 职责分层 | ST 框架在 `test/frame/`；算子 helper 放 `test/<op>/`；legacy 工具留 `test/utils/` |
| arch 目录 | SOC 相关测试源文件放 `arch35/` 等，CSV 同样放 arch 目录 |

---

## 2 参考实现对照

| 项目 | gbmv | stpttr |
|------|------|--------|
| 测试源文件 | `test/gbmv/arch35/gbmv_test.cpp` | `test/stpttr/arch35/stpttr_test.cpp` |
| Golden | `test/gbmv/arch35/gbmv_golden.h` | `test/stpttr/arch35/stpttr_golden.h` |
| CSV | `test/gbmv/arch35/gbmv_testcases.csv` | `test/stpttr/arch35/stpttr_testcases.csv` |
| 指针模型 | device（A/x/y） | device（AP/A） |
| 精度 | MERE_MARE（CSV） | EXACT（代码硬编码） |
| 异常测试 | `expect_success=false` 简单校验 | 按 `case_id` 细分错误码 |
| CMake | `ops_blas_add_gtest_tests(${OPS_BLAS} gbmv_test)` | 同左，`stpttr_test` |

---

## 3 风险与注意事项

1. **禁止每用例 aclInit**：必须在 `SetUpTestSuite` 级管理 ACL。
2. **负步长**：先读 kernel 索引方式；gbmv kernel 内部处理负 inc，测试传 buffer 基址；golden/verify 用 `verifyStridedVector` 对齐。
3. **const 与 device 指针**：device 路径常需 `(float*)` / `const_cast` 传给 ACL API，仅在测试代码中使用。
4. **case_id 命名**：避免 GTest 不支持的字符；`-` 会被 `gtest.h` 替换为 `_`。
5. **expect_success=false**：不跑 golden；需明确期望的 `aclblasStatus_t`。
6. **CSV 与代码同步**：新增 CSV 列需同时扩展 `config.h` 的解析逻辑。
7. **SOC arch 目录**：若算子仅在 `arch35/` 下有测试 cpp，CSV 也应放在对应 arch 目录以便 POST_BUILD 复制。

---

## 4 新算子 ST 开发步骤

### 4.1 准备环境

```bash
source <cann>/set_env.sh
# 确认 test/frame/ 下 6 个 ST 框架头文件存在
# 确认目标 SOC 在 build.sh --soc 支持列表中
```

### 4.2 分析 API 契约

阅读 `include/cann_ops_blas.h` 与 `blas/<op>/*_host.cpp`，明确：

- [ ] 入参中哪些是指针（A/x/y/AP 等），均为 device 地址
- [ ] kernel 如何处理负步长（基址 vs BLAS 调整指针）
- [ ] 标量 alpha/beta 是否 host 栈变量
- [ ] n=0 / m=0 等边界行为
- [ ] 推荐精度模式（Level-1 ABS、Level-2 MERE_MARE、拷贝类 EXACT）

### 4.3 编写 `<op>_testcases.csv`

1. 首行列头对齐 `TestCaseConfig` 字段名。
2. L0：小规模典型功能 + 关键边界。
3. L1：步长、特殊值、大规模、错误入参（`expect_success=false`）。
4. 精度列：浮点算子填 `mere_threshold`；整数/拷贝类在代码中设 EXACT。

### 4.4 实现 `<op>_golden.h`

- CPU 参考实现，接口建议：`(const TestCaseConfig&, inputs..., std::vector<float>& out)`
- 枚举解析（`parseTrans`、`parseUplo`）可放在 golden 头内
- 与 NPU 结果可比：同一套输入数据、同一套存储布局

### 4.5 编写 `<op>_test.cpp`

推荐结构：

```cpp
// 1. includes: gtest, config.h, verify.h, gtest.h, device.h, <op>_golden.h
//    算子专用 helper 如 gbmv_test_utils.h

// 2. 匿名 namespace：算子专属 helper（指针解析、device 分配等）

// 3. TestWithParam fixture + SetUpTestSuite/TearDownTestSuite

// 4. load<Op>Cases()：ConfigLoader + 可选 verifyCfg 覆盖

// 5. INSTANTIATE_TEST_SUITE_P

// 6. TEST_P 主体：
//    if (!tc.expectSuccess) { /* 错误码 */ return; }
//    /* 准备数据 */
//    /* host 数据 → H2D → 调 API(device ptr) → sync → D2H → golden → verify */

// 7. custom main
```

参考 gbmv / stpttr 的 device 流程段落。

### 4.6 添加 CMakeLists.txt

```cmake
ops_blas_add_gtest_tests(${OPS_BLAS} <op>_test)
```

若测试源在 `arch35/` 下，保持 `test/<op>/CMakeLists.txt` 在算子根目录即可（cmake 会自动发现 arch 源文件并复制 arch 目录下 CSV）。

### 4.7 构建验收

```bash
bash build.sh --ops=<op> --soc=ascend950 --run
```

检查：

- [ ] 全部 `[  PASSED  ]`
- [ ] 失败用例的 Verifier 日志可定位 diff
- [ ] `build/test/<op>/` 目录含 CSV 与可执行文件

### 4.8 编写 `test/<op>/README.md`

建议包含：算子功能、API 签名、device 指针约定、CSV 列说明、精度模式、构建运行命令、用例数量。

---

## 5 文件清单（新算子交付）

| 文件 | 必需 | 说明 |
|------|------|------|
| `test/<op>/<op>_testcases.csv` | 是 | 用例参数表 |
| `test/<op>/<op>_test.cpp` 或 `arch35/<op>_test.cpp` | 是 | GTest 入口 |
| `test/<op>/<op>_golden.h` 或 `arch35/<op>_golden.h` | 是 | CPU golden |
| `test/<op>/CMakeLists.txt` | 是 | `ops_blas_add_gtest_tests` |
| `test/<op>/README.md` | 推荐 | 算子 ST 说明 |
| `test/<op>/<op>_perf_test.cpp` | 可选 | 性能测试（不走精度框架） |
| `test/<op>/<op>_config.json` | 否 | 可选；当前框架不强制 |

---

## 6 附录：gbmv 测试数据流

```
gbmv_testcases.csv
       ↓ ConfigLoader (config.h)
TestCaseConfig (含 verifyCfg)
       ↓ generateHostDataGbmv (gbmv_test_utils.h)
hostData = [A, x, y, y_initial, alpha, beta]
       ↓ allocAndCopyToDevice (device.h)
device buffers [dA, dX, dY]
       ↓ aclblasSgbmv(handle, ..., dA, dX, dY)   // device 基址，kernel 内部处理负步长
       ↓ aclrtSynchronizeStream → D2H(y)
outputHost  vs  gbmv_golden_impl(...)
       ↓ verifyGbmvResult → verifyStridedVector (MERE_MARE)
PASS / FAIL
```

## 7 附录：stpttr 测试数据流

```
stpttr_testcases.csv
       ↓ ConfigLoader + EXACT 硬编码
TestCaseConfig
       ↓ expect_success ? 分派 : case_id 错误码测试
makeStpttrApData(n) → aclrtMalloc/H2D
       ↓
aclblasStpttr(handle, uplo, n, dAP, dA, lda)   // device 指针
       ↓ sync + D2H
aHost  vs  stpttr_golden_impl(...)
       ↓ verifyDenseVector (EXACT, sentinel 区在 golden 中处理)
PASS / FAIL
```

---

