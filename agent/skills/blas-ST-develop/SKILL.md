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
| `fill.h` | `BlasFillMode` 结构体、`makeBlasArray`、`makeBlasTriangular`、`makeBlasBanded`、`makeBlasStrided`、`makeBlasMatrix` |
| `verify.h` | `Verifier` 精度比对类 |
| `types.h` | `VerifyConfig`、`PrecisionMode` |
| `data.h` | `DataGenerator`（旧式算子兼容用） |
| `device.h` | `DeviceBuffer`、`allocAndCopyToDevice`、`adjustStridedBase`（旧式算子兼容用） |

### BLAS/LAPACK 参考库（`test/utils/`）

| 头文件 | 职责 |
|--------|------|
| `cblas_compat.h` | `aclblas` 枚举到 CBLAS/LAPACK 枚举的映射函数（`ToCblasOp`/`ToCblasUplo`/`ToCblasDiag`），Fortran BLAS/LAPACK 函数声明（`srotm_`/`stpttr_`/`strttp_`/`sgeqrf_`/`sgetrf_`） |

**依赖**：CMake 自动链接 `libblas`（OpenBLAS）和 `liblapack`，golden 文件可直接调用 `cblas_*` 和 Fortran BLAS/LAPACK 函数。

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
| Level-3 浮点（gemm） | MERE_MARE | 同上 |
| 矩阵分解（getrf/geqrf） | MERE_MARE | 同上 |

**PrecisionMode 选择与配置示例**：

```cpp
// 默认模式（无 CSV 自定义列，param 字段或 TEST_P 内固定）
VerifyConfig cfg;
cfg.mode = PrecisionMode::ABS;
cfg.absThreshold = 1e-5f;
EXPECT_TRUE(Verifier::verifyVector(npuResult, golden.data(), n, 1, cfg, p.caseName));

// MERE_MARE 模式（带 CSV 自定义列 mere_threshold / mare_multiplier，用例级控制精度）
// CSV: case_name,...,mere_threshold,mare_multiplier,expect_result
//       TC_01,...,1e-13,10,ACLBLAS_STATUS_SUCCESS
VerifyConfig cfg;
cfg.mode = PrecisionMode::MERE_MARE;
cfg.mereThreshold = p.mereThreshold;        // 从 CSV 读取
cfg.mareMultiplier = p.mareMultiplier;      // 从 CSV 读取
EXPECT_TRUE(Verifier::verifyVector(npuResult, golden.data(), n, stride, cfg, p.caseName));

// 精度模式需在 param.h 中扩展字段（若使用 CSV 自定义列）
struct SxParam : public BlasTestParamBase {
    // ... API 常规字段 ...
    float mereThreshold = 1e-5f;
    float mareMultiplier = 1.0f;
    SxParam(const csv_map& m) : BlasTestParamBase(m) {
        mereThreshold = parseFloat(ReadMap(m, "mere_threshold", "1e-5"));
        mareMultiplier = parseFloat(ReadMap(m, "mare_multiplier", "1.0"));
        // ...
    }
};
```

> **强制**：`VerifyConfig.mode` 必须显式设置，不得依赖默认值（默认行为由 `Verifier` 类构造器决定，可能与算子精度需求不匹配）。

---

## 第 2 步：编写 Param

文件： `test/{op}/{op}_param.h`

继承 `BlasTestParamBase`，字段按 API 参数顺序排列。数组参数类型为 `BlasFillMode`。

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
    BlasFillMode ap = BlasFillMode("INDEX");          // 顺序正整数 1, 2, 3, ...
    BlasFillMode a  = BlasFillMode("VALUE_NORM_N999"); // 哨兵值 -999
    int lda = 0;

    StpttrParam(const csv_map& m) : BlasTestParamBase(m) {
        uplo = parseFillMode(ReadMap(m, "uplo", "LOWER"));
        n    = parseInt(ReadMap(m, "n", "0"));
        ap   = BlasFillMode(ReadMap(m, "ap", "INDEX"));
        a    = BlasFillMode(ReadMap(m, "a", "VALUE_NORM_N999"));
        lda  = parseInt(ReadMap(m, "lda", std::to_string(std::max(1, n))));
    }
};

#endif
```

**BlasFillMode 命名规则**：`METHOD_PATTERN_VAL...`，从某位开始可不填（后续取默认值），不允许跳位。参数化 PATTERN（如 BANDED）先消耗结构参数，剩余 VAL 用于填充值。

**⚠️ RANDOM 值域约束**：使用 `RANDOM` 方法时，**必须**在 BlasFillMode 中显式指定值域（如 `RANDOM_NORM_1E6`），禁止使用裸 `RANDOM`。若无法确定值域，必须发送问卷向用户确认，默认值域为 `RANDOM_NORM_1`。

| 位 | 可选值 | 说明 |
|----|--------|------|
| METHOD（必填） | `NULLPTR` / `INDEX` / `RANDOM` / `VALUE` | 值获取方式 |
| PATTERN | `NORM` / `UPPER` / `LOWER` / `DIAG` / `ALTER` / `EXTREME` / `ILLCOND` / `BANDED` | 矩阵形状/分布模式 |
| VAL... | 数值（`N`前缀=负，`P`可省略）或特殊标记 | 结构参数 + 填充值参数 |

**常用写法示例**：

| CSV 写法 | 含义 |
|---------|------|
| `INDEX` | 顺序 1, 2, 3, ... |
| `INDEX_NORM_N1` | 顺序 -1, -2, -3, ... |
| `INDEX_ALTER` | 正负交替 |
| `RANDOM_1_3` | 随机 [−1, 3]（**推荐显式指定范围**） |
| `RANDOM_NORM_1E6` | 随机 [−1e6, 1e6]（带 NORM 模式） |
| `RANDOM_UPPER_0.5_2.0` | 上三角随机 [−0.5, 2.0] |
| `RANDOM_LOWER` | 下三角随机（**禁止**省略范围，应改为 `RANDOM_LOWER_1_10`） |
| `RANDOM_DIAG_1_100` | 对角随机 [−1, 100] |
| `RANDOM_BANDED_2_3_N5_5` | 带状矩阵 kl=2 ku=3，band 内随机 [−5, 5] |
| `INDEX_BANDED_1_1` | 带状矩阵 kl=1 ku=1，band 内顺序值 |
| `VALUE_BANDED_2_2_0` | 带状矩阵 kl=2 ku=2，band 内全零 |

> **强制（reviewer HIGH）**：CSV 中所有 `RANDOM` 必须显式指定值域范围（`RANDOM_lo_hi` 或 `RANDOM_PATTERN_lo_hi`），**禁止**仅写 `RANDOM` 依赖默认范围（默认范围为 [−FLT_MAX, FLT_MAX]，不可控）。

| `VALUE_NORM_0` | 全零 |
| `VALUE_NORM_1` | 全一 |
| `VALUE_NORM_N999` | 哨兵值 -999 |
| `VALUE_NORM_1E10` | 大常数 1e10 |
| `VALUE_NORM_INF` | 正无穷 |
| `VALUE_NORM_NAN` | 非数 |
| `VALUE_DIAG_1` | 单位矩阵 |

**BlasTestParamBase 公共字段**：`caseName`（用例名）、`description`（语义描述）、`expectResult`（期望返回码）、`randomSeed`（随机种子，通过 CSV 列 `random_seed` 传入，默认 `0`，用于 RNG 可复现性）。

---

## 第 3 步：编写 cpu / npu

### 3.1 cpu.h（golden）

文件： `test/{op}/{op}_golden.h`

签名与 BLAS API **完全一致**，返回 `aclblasStatus_t`。**保留参数校验**（与 NPU 算子保持一致），校验通过后调用参考库函数。

#### CBLAS 算子示例（Level-1/2/3）

```cpp
#include "cblas_compat.h"

inline aclblasStatus_t aclblasSgemv_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || lda < std::max(1, m)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    cblas_sgemv(CblasColMajor, ToCblasOp(trans), m, n, *alpha, a, lda, x, incx, *beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
```

#### LAPACK 算子示例（geqrf/getrf/tpttr/trttp）

```cpp
#include "cblas_compat.h"

inline aclblasStatus_t aclblasStpttr_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ap == nullptr || a == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    char uploChar = (uplo == ACLBLAS_UPPER) ? 'U' : 'L';
    int info = 0;
    stpttr_(&uploChar, &n, ap, a, &lda, &info, 1);
    return ACLBLAS_STATUS_SUCCESS;
}
```

**关键约定**：
- 使用 `cblas_compat.h` 提供的 `ToCblasOp`/`ToCblasUplo`/`ToCblasDiag` 转换枚举
- Fortran 函数声明已在 `cblas_compat.h` 中，直接调用即可
- 保留参数校验，与 NPU 算子保持一致（CBLAS 对空指针会崩溃，必须校验）
- 使用 `CblasColMajor`（列主序），与 BLAS 标准一致

### 3.2 npu.h（device wrapper）

文件： `test/{op}/arch35/{op}_npu_wrapper.h`

封装 ACL 准备和释放工作，入参 `nullptr` 时跳过对应 device 内存操作。**每个 ACL 调用必须校验返回值**（`aclrtMalloc` / `aclrtMemcpy` H2D / `aclrtMemcpy` D2H / `aclrtSynchronizeDevice` / `aclrtFree`），失败时立即清理并返回错误码。

```cpp
// test/{op}/arch35/{op}_npu_wrapper.h — 完整模板（带返回值校验）
#ifndef {OP}_NPU_H
#define {OP}_NPU_H

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblas{Op}_npu(
    aclblasHandle_t handle, /* API 参数：维度 + const 指针 + 非常量指针 */)
{
    // 1. 快速路径：handle == nullptr 或 n <= 0 → 直接透传（由算子内部处理）
    if (handle == nullptr || n <= 0) {
        return aclblas{Op}(handle, /* 参数 */);
    }

    // 2. 计算 host 端需要搬运的字节数（考虑 stride / lda / 多维）
    const size_t xBytes = /* ... */;
    const size_t yBytes = /* ... */;

    // 3. 分配 device 内存 + H2D（每个 malloc/H2D 必须校验返回值）
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dX);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (y != nullptr) {
        aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (dX) aclrtFree(dX);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (dX) aclrtFree(dX);
            aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // 4. 调用算子（必须校验返回状态）
    aclblasStatus_t ret = aclblas{Op}(handle, /* 参数转换为 device 指针 */);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        return ret;
    }

    // 5. 同步设备（必须校验返回值）
    aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // 6. D2H（必须校验返回值）+ 释放
    if (y != nullptr && dY != nullptr) {
        aclRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            if (dX) aclrtFree(dX);
            aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dX) aclrtFree(dX);
    if (dY) aclrtFree(dY);
    return ret;
}

#endif
```

**NPU wrapper 强制约束**（reviewer HIGH）：
- `aclrtSynchronizeDevice()` 返回 `ACL_SUCCESS` 必须校验
- `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)` D2H 返回 `ACL_SUCCESS` 必须校验
- 任何 ACL 失败后必须 **free 已分配的 device 内存** 再返回对应错误码（防止泄漏）
- wrapper 内部 **不得**调用业务算子的日志接口（如 `OP_LOGE`），只返回结构化错误码

**特殊场景**：
- 若算子是异步的（调用方需要流上同步），可改为 `aclrtSynchronizeStream(h->stream)` 并校验返回值
- 入参 `nullptr` 表示该 buffer 不参与（如 `beta==0` 时 y 无需 H2D 但需 D2H）

---

## 第 4 步：编写 CSV + GTest

### 4.1 CSV 用例表

文件： `test/{op}/arch35/{op}_test.csv`

列名 = API 参数名 + 框架公共列，顺序与接口声明一致。枚举值写 `ACLBLAS_` 完整前缀。

**框架公共列**：
| 列名 | 说明 | 必填 |
|------|------|------|
| `case_name` | 用例名（GTest 显示名） | 是 |
| `description` | 语义描述 | 否 |
| `expect_result` | 期望返回码（默认 `SUCCESS`） | 否 |
| `random_seed` | 随机种子（默认 `0`，用于 RNG 可复现性） | 否 |

```csv
case_name,description,uplo,n,ap,a,lda,expect_result,random_seed
TC_L0_01,handle_null,ACLBLAS_LOWER,5,NULLPTR,NULLPTR,5,ACLBLAS_STATUS_NOT_INITIALIZED,0
TC_L0_06,n1_lower,ACLBLAS_LOWER,1,INDEX,VALUE_NORM_N999,1,ACLBLAS_STATUS_SUCCESS,0
TC_L1_19,zeros_lower,ACLBLAS_LOWER,8,VALUE_NORM_0,VALUE_NORM_N999,8,ACLBLAS_STATUS_SUCCESS,0
TC_L1_20,random_lower,ACLBLAS_LOWER,8,RANDOM_NORM_1E6,RANDOM_NORM_1E6,8,ACLBLAS_STATUS_SUCCESS,1
```

- `expect_result`：`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_NOT_INITIALIZED`
- 数组列的值为 `BlasFillMode` 字符串（见上方命名规则）
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

    std::vector<float> apHost = makeBlasTriangular(p.n, p.uplo == ACLBLAS_UPPER, p.ap, p.randomSeed);
    std::vector<float> aHost  = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a, p.randomSeed);

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
| 数组填充不匹配 | 检查 `BlasFillMode` 字符串是否正确，三角矩阵用 `makeBlasTriangular`，带状用 `makeBlasBanded` |
| 精度 fail | 看 Verifier 日志中的 MERE/MARE 或 exact mismatch 计数 |
| `gtest_main` 链接冲突 | 框架统一使用 `test/frame/test_main.cpp`，勿自行写 `main()` |
