---
name: blas-ST-develop
description: |
  为 BLAS 算子开发 GTest + CSV 驱动的精度 ST。触发场景：
  - 新算子 ST、编写 xxx_param.h / xxx_golden.h / xxx_npu_wrapper.h / xxx_test.cpp / xxx_test.csv
  - 改写旧式 TEST_F 为 CSV 参数化测试
  按 6 步执行：分析 API → 写 param → 写 cpu/npu → 写 GTest → CMake → build.sh 验证。
metadata:
  author: wangzitao_leo
---

# BLAS 算子 ST 开发技能

## 概述

ops-blas 精度 ST 采用 **GTest 参数化 + CSV 用例表** 驱动，每个算子交付件为 5 个文件。

### 交付清单

```
test/{op}/
├── {op}_param.h            ← 参数结构体（与芯片无关）
├── {op}_golden.h           ← CPU golden（与芯片无关），签名与 BLAS API 一致
├── CMakeLists.txt
└── arch35/
    ├── {op}_npu_wrapper.h  ← NPU wrapper（芯片相关 ACL 操作）
    ├── {op}_test.cpp       ← GTest 入口：BlasTest<Param> + TEST_P 5 步流程
    └── {op}_test.csv       ← CSV 用例表，列名=API 参数名
```

### ST 框架头文件（`test/frame/`）

| 头文件 | 职责 |
|--------|------|
| `csv_loader.h` | `csv_map`、`ReadMap`、`GetCasesFromCsv`、`PrintCaseInfoString`、枚举解析、`BlasTestParamBase`、`isNullHandleCase`、`parseInt/parseFloat/parseDouble/parseUint` |
| `blas_test.h` | `BlasTest<ParamType>` 模板基类（含 SetUpTestSuite/TearDownTestSuite/handle_/stream_） |
| `fill.h` | `BlasDataFill` 枚举、`makeBlasArray`、`makeBlasTriangular`、`makeBlasBanded`、`makeBlasStrided`、`applyExoticFill` |
| `verify.h` | `Verifier` 精度比对类 |
| `types.h` | `VerifyConfig`、`PrecisionMode` |
| `data.h` | `DataGenerator`（旧式算子兼容用） |
| `device.h` | `DeviceBuffer`、`allocAndCopyToDevice`、`adjustStridedBase`（旧式算子兼容用） |

---

## 前置条件

```bash
source <CANN>/set_env.sh
cd ops-blas
ls test/frame/csv_loader.h test/frame/blas_test.h test/frame/fill.h test/frame/verify.h
```

---

## 第 1 步：分析 API

阅读 `include/cann_ops_blas.h` 与算子实现代码，确认 API 签名与精度模式。

### 1.1 数据指针约定

**NPU wrapper（`_npu_wrapper.h`）封装全部 ACL 操作** —— 测试侧只需准备 host 端 `std::vector<float>` 并传入 `_npu`。`_npu` 内部完成 malloc → H2D → kernel → sync → D2H → free。若入参为 `nullptr`，`_npu` 跳过分配直接透传。

### 1.2 精度模式

| 算子类型 | 推荐模式 | 配置方式 |
|----------|----------|----------|
| Level-2 浮点（gbmv） | MERE_MARE | param 字段 `mereThreshold` / `mareMultiplier`，CSV 列 `mere_threshold` / `mare_multiplier` |
| 格式转换 / pack-unpack | EXACT | 在 `TEST_P` 内设 `cfg.mode = PrecisionMode::EXACT` |
| Level-1 向量 | ABS | 在 `TEST_P` 内设 `cfg.mode = PrecisionMode::ABS` |

---

## 第 2 步：编写 Param

文件： `test/{op}/{op}_param.h`

继承 `BlasTestParamBase`，字段按 API 参数顺序排列。数组参数类型为 `BlasDataFill`。

```cpp
#ifndef STPTTR_PARAM_H
#define STPTTR_PARAM_H

#include <string>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct StpttrParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int n = 0;
    BlasDataFill ap = BlasDataFill::INDEX;
    BlasDataFill a  = BlasDataFill::SENTINEL;
    int lda = 0;

    StpttrParam(const csv_map& m) : BlasTestParamBase(m) {
        uplo = parseFillMode(ReadMap(m, "uplo", "LOWER"));
        n    = parseInt(ReadMap(m, "n", "0"));
        ap   = parseDataFill(ReadMap(m, "ap", "index"));
        a    = parseDataFill(ReadMap(m, "a", "sentinel"));
        lda  = parseInt(ReadMap(m, "lda", std::to_string(std::max(1, n))));
    }
};

#endif
```

**BlasDataFill 取值**：`index`、`random`、`zeros`、`ones`、`nullptr`、`sentinel`

**BlasTestParamBase 公共字段**：`caseName`（用例名）、`description`（语义描述）、`expectResult`（期望返回码）

---

## 第 3 步：编写 cpu / npu

### 3.1 cpu.h（golden）

文件： `test/{op}/{op}_golden.h`

签名与 BLAS API **完全一致**，返回 `aclblasStatus_t`，包含完整参数校验和纯 CPU 计算逻辑。

```cpp
inline aclblasStatus_t aclblasStpttr_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
    // ... CPU 拆包逻辑 ...
    return ACLBLAS_STATUS_SUCCESS;
}
```

### 3.2 npu.h（device wrapper）

文件： `test/{op}/arch35/{op}_npu_wrapper.h`

封装 ACL 准备和释放工作，入参 `nullptr` 时跳过对应 device 内存操作。

```cpp
inline aclblasStatus_t aclblasStpttr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStpttr(handle, uplo, n, ap, a, lda);
    }
    // ap != nullptr → aclrtMalloc(dP) + H2D
    // a  != nullptr → aclrtMalloc(dA) + H2D
    // kernel → sync → D2H → free
    return ret;
}
```

---

## 第 4 步：编写 CSV + GTest

### 4.1 CSV 用例表

文件： `test/{op}/arch35/{op}_test.csv`

列名 = API 参数名，顺序与接口声明一致。枚举值写 `ACLBLAS_` 完整前缀。

```csv
case_name,description,uplo,n,ap,a,lda,expect_result
TC_L0_01,handle_null,ACLBLAS_LOWER,5,nullptr,nullptr,5,ACLBLAS_STATUS_NOT_INITIALIZED
TC_L0_06,n1_lower,ACLBLAS_LOWER,1,index,sentinel,1,ACLBLAS_STATUS_SUCCESS
TC_L1_19,zeros_lower,ACLBLAS_LOWER,8,zeros,sentinel,8,ACLBLAS_STATUS_SUCCESS
```

- `expect_result`：`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_NOT_INITIALIZED`
- 数组列的值为 `BlasDataFill` 枚举
- CSV 路径由 `ReplaceFileExtension2Csv(__FILE__)` 自动推导（与 .cpp 同名 .csv）

### 4.2 GTest 入口

文件： `test/{op}/arch35/{op}_test.cpp`

```cpp
class StpttrArch35Test : public BlasTest<StpttrParam> { };

TEST_F(StpttrArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasStpttr_npu(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Stpttr, StpttrArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StpttrParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StpttrParam>);

TEST_P(StpttrArch35Test, CsvDriven) {
    const auto& p = GetParam();

    std::vector<float> apHost = makeBlasTriangular(p.n, p.uplo == ACLBLAS_UPPER, p.ap, p.description);
    std::vector<float> aHost  = makeBlasArray(static_cast<size_t>(p.lda) * p.n, p.a);

    const float* apPtr = apHost.empty() ? nullptr : apHost.data();
    float*       aPtr  = aHost.empty()  ? nullptr : aHost.data();

    aclblasStatus_t ret = aclblasStpttr_npu(StpttrArch35Test::handle_, p.uplo, p.n, apPtr, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    std::vector<float> golden(aHost.size());
    aclblasStpttr_cpu(StpttrArch35Test::handle_, p.uplo, p.n, apHost.data(), golden.data(), p.lda);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
```

**5 步测试流程**：生成数据 → `_npu` 执行 → 失败比对错误码 → `_cpu` 算 golden → `Verifier::verifyVector` 比对

**关键约定**：
- null handle 测试用 `TEST_F` 单独测，不下 CSV
- `_npu` 已处理 n<=0 和 nullptr 透传，测试侧无需特殊分支
- 测试文件不写 `main()`，由 `test/frame/test_main.cpp` 统一提供

---

## 第 5 步：CMake

```cmake
# test/{op}/CMakeLists.txt
ops_blas_add_gtest_tests(${OPS_BLAS} <op>_test)
```

CMake 自动发现 `test/<op>/arch35/<op>_test.cpp`，`test/frame/test_main.cpp` 作为共享 main 入口。

---

## 第 6 步：构建验证

```bash
source <CANN>/set_env.sh
cd ops-blas
bash build.sh --ops=stpttr --run              # 默认卡0
bash build.sh --ops=stpttr --run --device=1   # 指定卡1
```

通过标准：`[  PASSED  ] N tests.`，Summary 中 `Failed: 0`。

---

## 常见问题

| 现象 | 处理 |
|------|------|
| CSV 读取失败 | 确认 CSV 与 .cpp 同名同目录，`ReplaceFileExtension2Csv(__FILE__)` 自动定位 |
| null handle 测试多余代码 | 改用 `TEST_F` 单独测，不下 CSV |
| 数组填充不匹配 | 检查 `BlasDataFill` 取值是否正确，三角矩阵用 `makeBlasTriangular`，带状用 `makeBlasBanded` |
| 精度 fail | 看 Verifier 日志中的 MERE/MARE 或 exact mismatch 计数 |
| `gtest_main` 链接冲突 | 框架统一使用 `test/frame/test_main.cpp`，勿自行写 `main()` |

---

## 已验证算子

| 算子 | 指针 | 精度 | CSV 行数 | SOC |
|------|------|------|----------|-----|
| gbmv | device (A/x/y) | MERE_MARE | 51 | ascend950 |
| stpttr | device (AP/A) | EXACT | 55 | ascend950 |
| strttp | device (A/AP) | EXACT | 55 | ascend950 |
