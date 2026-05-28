---
name: blas-ST-develop
description: |
  为 BLAS 算子开发 GTest + CSV 驱动的精度 ST。触发场景：
  - 新算子 ST、适配统一测试框架、创建 xxx_testcases.csv / xxx_golden.h / xxx_test.cpp
  - 改写旧式 TEST_F 为 CSV 参数化测试
  按 6 步执行：分析 API → 写 CSV → 写 golden → 写 GTest → CMake → build.sh 验证。
metadata:
  author: wangzitao_leo
---

# BLAS 算子 ST 开发技能

## 概述

ops-blas 精度 ST 采用 **GTest 参数化 + CSV 用例表 + 测试代码内硬编码算子逻辑**。

- 用例参数：`<op>_testcases.csv`
- 算子逻辑：`<op>_test.cpp` + `<op>_golden.h`
- ST 框架头文件：`test/frame/`（6 个头文件，见下表）
- 算子专用 helper：放 `test/<op>/`（如 `gbmv_test_utils.h`）
- 旧版工具：`test/utils/error_check.h`、`test/utils/golden.h`（perf / legacy 测试用，与 ST 框架无关）
- **不需要** `*_config.json`（可选，当前 gbmv / stpttr / strttp 均未使用）

详细设计见 `docs/zh/develop/st_develop_guide.md`。

### ST 框架头文件（`test/frame/`，共 6 个）

| 头文件 | 职责 |
|--------|------|
| `types.h` | `TestCaseConfig`、`OpConfig`、`VerifyConfig`、`PrecisionMode` |
| `config.h` | `ConfigLoader`：CSV / JSON 用例加载 |
| `verify.h` | `Verifier` 精度引擎 + `verifyDenseVector` / `verifyStridedVector` |
| `device.h` | `DeviceBuffer`、`allocAndCopyToDevice`、`adjustStridedBase` |
| `data.h` | `DataGenerator` 矩阵/向量填充原语 |
| `gtest.h` | GTest 参数名生成（`gtestParamNameFromTestCase`） |

## 前置条件

```bash
source <CANN>/set_env.sh
cd ops-blas
ls test/frame/types.h test/frame/config.h test/frame/verify.h test/frame/gtest.h
```

---

## 第 1 步：分析 API

阅读 `include/cann_ops_blas.h` 与 `blas/<op>/*_host.cpp`，确认 API 契约与精度模式。

### 1.1 数据指针约定（ST 框架默认 device）

**ST 框架统一假设 API 的数据指针（A/x/y/AP 等）指向 device 内存。** 测试标准流程：

```
host 生成数据 → allocAndCopyToDevice / aclrtMalloc + H2D → 调 API → sync → D2H → golden 比对
```

| 要点 | 说明 |
|------|------|
| 传参 | 传 **device buffer 基址**，不要传 BLAS 调整后的指针（除非 host 层明确要求） |
| 负步长 | 部分 kernel（如 gbmv）在内部用 `(dim-1-i)*(-inc)` 索引，测试侧只传基址 + inc 值 |
| 标量 alpha/beta | 仍为 host 栈变量，取地址后传入 |

参考算子：gbmv（A/x/y）、stpttr（AP/A）、strttp（A/AP）。

### 1.2 精度模式

| 算子类型 | 推荐模式 | 配置方式 |
|----------|----------|----------|
| Level-2 浮点（gbmv） | MERE_MARE | CSV 列 `mere_threshold`、`mare_multiplier` |
| 格式转换 / pack-unpack | EXACT | 在 `load*Cases()` 中设 `tc.verifyCfg.mode = PrecisionMode::EXACT` |
| Level-1 向量 | ABS | CSV 列 `abs_tol` 或 load 时硬编码 |

### 1.3 数据布局（写在 golden / test 里，非配置文件）

| 布局 | DataGenerator / golden |
|------|------------------------|
| banded | `fillBandedMatrix` |
| packed 三角 | pack/unpack 逻辑（见 strttp / stpttr golden） |
| dense + stride | golden 与 kernel 索引方式保持一致；验证用 `verifyStridedVector` |

---

## 第 2 步：创建 CSV

文件： `test/<op>/<op>_testcases.csv` 或 `test/<op>/arch35/<op>_testcases.csv`

首行列头对齐 `types.h` 中 `TestCaseConfig` 字段（snake_case）。算子只填用到的列。

常用列：

```
case_id,level,description,uplo,n,lda,trans,m,kl,ku,alpha_real,beta_real,incx,incy,seed,expect_success,mere_threshold,mare_multiplier
```

约定：

- `expect_success=true` → 正常路径 + golden 比对
- `expect_success=false` → 仅校验错误码（在 `TEST_P` 中按 `case_id` 分派）
- `description` 可携带语义（如 `zeros_lower`、`roundtrip_lower`），供 test 解析特殊值
- `case_id` 含 `-` 时，用 `gtest.h` 的 `gtestParamNameFromTestCase` 自动替换为 `_`

---

## 第 3 步：实现 golden

文件： `test/<op>/<op>_golden.h`（可与 test 同目录或 `arch35/`）

要点：

1. CPU 参考实现，接口如 `<op>_golden_impl(tc, inputs..., goldenOutput)`
2. 枚举解析（`parseTrans`、`parseUplo`）放在 golden 头内
3. 负步长：与 kernel 索引方式对齐；验证阶段用 `verifyStridedVector` 处理负 stride
4. 列优先：`A[i + j*lda]`；banded：`A[(ku+i-j) + j*lda]`

---

## 第 4 步：编写 GTest 入口

文件： `test/<op>/<op>_test.cpp` 或 `test/<op>/arch35/<op>_test.cpp`

### 4.1 最小模板

```cpp
#include <gtest/gtest.h>
#include "config.h"
#include "verify.h"
#include "gtest.h"
#include "device.h"   // 需要 H2D/D2H 时
#include "<op>_golden.h"
// 算子专用：如 gbmv 额外 include "gbmv_test_utils.h"

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
    static void TearDownTestSuite() {
        aclrtDestroyStream(stream_);
        aclblasDestroy(handle_);
        aclrtResetDevice(0);
        aclFinalize();
    }
    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

static std::vector<TestCaseConfig> loadXxxCases() {
    auto [cases, cfg] = ConfigLoader::loadAllForOp(g_configDir, "<op>");
    (void)cfg;
    // 若用 EXACT：for (auto& tc : cases) tc.verifyCfg.mode = PrecisionMode::EXACT;
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Xxx, XxxTest,
    ::testing::ValuesIn(loadXxxCases()),
    gtestParamNameFromTestCase);

TEST_P(XxxTest, CsvDriven) {
    const TestCaseConfig& tc = GetParam();
    if (!tc.expectSuccess) { /* 按 case_id 断言错误码 */ return; }
    if (n == 0) { /* 边界 SUCCESS */ return; }
    /* 特殊用例：roundtrip 等 */
    /* 正常路径：host 数据 → H2D → API(device ptr) → sync → D2H → golden → verify */
}

int main(int argc, char* argv[]) {
    g_configDir = (argc > 1) ? argv[1] : ".";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 4.2 gbmv 参考流程（device 指针）

```cpp
auto hostData = generateHostDataGbmv(tc);
auto devBufs = allocAndCopyToDevice({hostData[0], hostData[1], hostData[2]});
// kernel 内部处理负步长，传 buffer 基址即可
aclblasSgbmv(handle_, transOp, m, n, kl, ku, &alpha,
    (const float*)devBufs[0]->ptr(), lda,
    (const float*)devBufs[1]->ptr(), incx,
    &beta, (float*)devBufs[2]->ptr(), incy);
aclrtSynchronizeStream(stream_);
devBufs[2]->copyToHost(outputHost.data(), outputHost.size() * sizeof(float));
verifyGbmvResult(tc, outputHost, goldenOutput);
```

### 4.3 必须遵守

| 规则 | 说明 |
|------|------|
| 自定义 `main()` | **不链接 `gtest_main`**；`argv[1]` 为 CSV 目录 |
| ACL 只初始化一次 | `SetUpTestSuite` / `TearDownTestSuite`；禁止每用例 `aclInit` |
| 标量取地址 | `float alpha = tc.alphaReal.value_or(1.0f);` 再 `&alpha` |
| 异常用例 | `expect_success=false` 在 `TEST_P` 内按 `case_id` 分支，不走 golden |
| n=0 成功路径 | 单独分支，勿对 size=0 做 `aclrtMalloc` |
| device 指针 | 禁止把 host `std::vector::data()` 直接传入 kernel 型 API |

算子专用逻辑写在 `test/<op>/` 下，不必放入 `test/frame/`。

---

## 第 5 步：CMake

```cmake
# test/<op>/CMakeLists.txt
ops_blas_add_gtest_tests(${OPS_BLAS} <op>_test)
```

- 自动发现 `test/<op>/<op>_test.cpp` 或 `arch35/<op>_test.cpp`
- `cmake/test.cmake` 已添加 `test/frame` 与 `test/utils` 到 include 路径
- POST_BUILD 自动复制 CSV 到可执行文件目录

---

## 第 6 步：构建验证

```bash
source <CANN>/set_env.sh
cd ops-blas
bash build.sh --ops=<op> --soc=ascend950 --run
```

通过标准：`[  PASSED  ] N tests.`，Summary 中 `Failed: 0`。

多算子：`bash build.sh --ops=gbmv,stpttr,strttp --soc=ascend950 --run`

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `Cannot open file: *_testcases.csv` | CSV 与 exe 同目录；确认 `argv[1]` 来自 build.sh |
| `gtest_main` 链接冲突 | 只用 `ops_blas_add_gtest_tests`，勿手动链 `gtest_main` |
| n=0 失败 | `TEST_P` 内单独处理，API 允许 nullptr |
| 负步长结果全错 | 确认是否误做了 BLAS 基指针调整；读 kernel 源码确认索引方式 |
| device 算子段错误 | 确认 H2D 后再调 API，勿传 host 指针 |
| 精度 fail | 看 Verifier 日志中的 MERE/MARE 或 exact mismatch 计数 |

---

## 已验证算子

| 算子 | 指针 | 精度 | 用例数 | SOC |
|------|------|------|--------|-----|
| gbmv | device (A/x/y) | MERE_MARE（CSV） | 51 | ascend950 |
| stpttr | device (AP/A) | EXACT（代码） | 59 | ascend950 |
| strttp | device (A/AP) | EXACT（代码） | 59 | ascend950 |

---

## 交付清单

| 文件 | 必需 |
|------|------|
| `<op>_testcases.csv` | 是 |
| `<op>_test.cpp` | 是 |
| `<op>_golden.h` | 是 |
| `CMakeLists.txt`（`ops_blas_add_gtest_tests`） | 是 |
