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
| `RANDOM` | 全值域随机 |
| `RANDOM_NORM_1E6` | 随机 [-1e6, 1e6] |
| `RANDOM_UPPER` | 上三角随机 |
| `RANDOM_LOWER` | 下三角随机 |
| `RANDOM_DIAG` | 对角随机 |
| `RANDOM_BANDED_2_3` | 带状矩阵 kl=2 ku=3，band 内随机 |
| `INDEX_BANDED_1_1` | 带状矩阵 kl=1 ku=1，band 内顺序值 |
| `VALUE_BANDED_2_2_0` | 带状矩阵 kl=2 ku=2，band 内全零 |
| `VALUE_NORM_0` | 全零 |
| `VALUE_NORM_1` | 全一 |
| `VALUE_NORM_N999` | 哨兵值 -999 |
| `VALUE_NORM_1E10` | 大常数 1e10 |
| `VALUE_NORM_INF` | 正无穷 |
| `VALUE_NORM_NAN` | 非数 |
| `VALUE_DIAG_1` | 单位矩阵 |

**BlasTestParamBase 公共字段**：`caseName`（用例名）、`description`（语义描述）、`expectResult`（期望返回码）

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
TC_L0_01,handle_null,ACLBLAS_LOWER,5,NULLPTR,NULLPTR,5,ACLBLAS_STATUS_NOT_INITIALIZED
TC_L0_06,n1_lower,ACLBLAS_LOWER,1,INDEX,VALUE_NORM_N999,1,ACLBLAS_STATUS_SUCCESS
TC_L1_19,zeros_lower,ACLBLAS_LOWER,8,VALUE_NORM_0,VALUE_NORM_N999,8,ACLBLAS_STATUS_SUCCESS
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
