# aclblasLtMatrixTransform

矩阵变换算子，提供矩阵的转置、缩放、加法以及多种内存布局（order）与数据类型之间的转换能力。采用 aclBLASLt 描述符风格接口，复用已有的 `aclblasLtHandle_t` / `aclblasLtMatrixLayout_t` 基础设施，并新增独立的变换描述符 `aclblasLtMatrixTransformDesc_t`。

## 计算语义

```
C = alpha · op(A) + beta · op(B)
```

- `op(·)` 为变换操作，由变换描述符的 `TRANSA` / `TRANSB` 属性指定：
  - `N`（ACLBLAS_OP_N）：原样，不转置。
  - `T`（ACLBLAS_OP_T）：转置。
  - `C`（ACLBLAS_OP_C）：共轭转置；本算子无复数数据类型，`C` 等价于 `T`。
- `alpha` / `beta` 为主机侧标量缩放系数（pointer mode = HOST），按变换描述符的 `scaleType`（计算精度类型）解释：浮点路径（FP32/FP16/BF16）与 FP8 路径为 `float`，整数路径为 `int32_t`，FP4 路径为 `bfloat16_t`（计算可再升 `float`）。
- 计算流程：输入按各自 dtype 读入，转换到 `scaleType` 域完成 `alpha·A + beta·B` 的缩放与求和，再转换为输出 dtype 并按 `Cdesc` 指定的 order 写回（`输入dtype → scaleType → 输出dtype`）。其中 FP8 仅经 `float`、FP4 仅经 `bfloat16_t` 完成与其它 dtype 的转换。
- 当 `beta = 0` 且 `B = nullptr` 时退化为 `C = alpha · op(A)`；当 `alpha = 1` 且 `beta = 0` 时退化为纯布局/类型转换。
- 当 `rows = 0` 或 `cols = 0`（空矩阵）时直接返回 `ACLBLAS_STATUS_SUCCESS`（no-op）。

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 (A/B/C 各自独立) | FP32 / FP16 / BF16 / INT8 / INT32 / FP8_E4M3FN / FP8_E5M2 / FP4_E2M1 |
| 计算精度类型 (scaleType) | 浮点路径 FP32；整数路径 INT32；FP8 路径 FP32；FP4 路径 BF16（可升 FP32） |
| 跨 dtype 转换 | 同 scaleType 路径内任意 in → out（浮点路径内 FP32/FP16/BF16 互转；整数路径内 INT8/INT32 互转；FP8 经 float 与 FP32/FP16/BF16 互转、E4M3FN/E5M2 互转；FP4 经 bf16 与 BF16/FP32 互转） |
| 内存布局 order | COL / ROW / COL32 / COL4_4R2_8C / COL32_2R_4R4 |
| 变换 op | N / T（C 等价 T） |
| 目标芯片 | Ascend950（Ascend950PR / Ascend950DT） |
| 目标架构 | arch35（DAV_3510） |

**数据类型明细**（A / B / C 各自 dtype 独立可设）：

| 数据类型 | aclDataType | C++ 类型 | 字节 | 计算路径 (scaleType) |
|---------|-------------|----------|------|----------------------|
| FP32 | ACL_FLOAT | float | 4 | 浮点（FP32） |
| FP16 | ACL_FLOAT16 | half | 2 | 浮点（FP32） |
| BF16 | ACL_BF16 | bfloat16_t | 2 | 浮点（FP32） |
| INT8 | ACL_INT8 | int8_t | 1 | 整数（INT32） |
| INT32 | ACL_INT32 | int32_t | 4 | 整数（INT32） |
| FP8_E4M3FN | ACL_FLOAT8_E4M3FN（=36） | float8_e4m3_t | 1 | FP8（FP32，cast 经 float） |
| FP8_E5M2 | ACL_FLOAT8_E5M2（=35） | float8_e5m2_t | 1 | FP8（FP32，cast 经 float） |
| FP4_E2M1 | ACL_FLOAT4_E2M1（=40） | float4_e2m1x2_t | 0.5（2 元素/字节，packed） | FP4（BF16，cast 经 bf16） |

> FP8 与 INT8 同为 1 字节/元素，搬运与复杂布局置换经字节级 reinterpret 复用整数路径的两步法机制（置换不改字节值）。FP4 为 `fp4x2` packed（2 个 4-bit 元素打包进 1 字节），最小可寻址单位为「列对」（2 个相邻元素），复杂布局置换走 bf16 解包/重打包路径（见下）。

**布局 order 说明**：

| order | 枚举值 | 含义 |
|-------|--------|------|
| COL | ACLBLASLT_ORDER_COL = 0 | 列主序 |
| ROW | ACLBLASLT_ORDER_ROW = 1 | 行主序 |
| COL32 | ACLBLASLT_ORDER_COL32 = 2 | 32 列为一组的复合 tile，组内列主序 |
| COL4_4R2_8C | ACLBLASLT_ORDER_COL4_4R2_8C = 3 | 32 列 × 8 行复合量化布局 |
| COL32_2R_4R4 | ACLBLASLT_ORDER_COL32_2R_4R4 = 4 | 32 列 × 32 行复合量化布局 |

**布局组合限制**：

- COL / ROW / COL32 支持全部 dtype（FP32/FP16/BF16/INT8/INT32/FP8/FP4）与全部 op（N/T，C 等价 T）。
- 复杂量化布局 COL4_4R2_8C / COL32_2R_4R4 仅支持量化路径 dtype：**INT8 / INT32 / FP8（E4M3FN/E5M2）/ FP4（E2M1）**，op 支持 N/T；纯浮点 dtype（FP32/FP16/BF16）与复杂量化布局的组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
  - INT8 / INT32 在 INT32 计算域，FP8 在 FP32 计算域，二者均为 1 字节/元素、在字节域做 tile 内置换；
  - FP4 在 BF16 计算域，packed 元素无法字节级独立置换，故复杂布局走「`fp4x2 → bf16` 解包 → bf16 域置换（与 INT8/FP8 复用同一置换规则）→ `bf16 → fp4x2` 重打包」路径，置换不改数值。
- 不支持：
  - FP64、复数（complex64/complex128）；
  - HiF8（ACL_HIFLOAT8）、FP8_E8M0、FP6、FP4_E1M2（首批仅做 FP8 E4M3FN/E5M2 + FP4 E2M1）；
  - 纯浮点 dtype（FP32/FP16/BF16）与复杂量化布局（COL4_4R2_8C / COL32_2R_4R4）的组合；
  - 跨 scaleType 路径转换（如浮点输入直接转整数输出、FP8 输入直接转 INT8 输出等跨浮点/整数/FP8/FP4 路径）；
  - batchCount > 1。

**leading dimension（ld，元素数）约束**（A/B/C 各自的 layout 描述符指定）：

| order | ld 下界 |
|-------|---------|
| COL | ld ≥ rows |
| ROW | ld ≥ cols |
| COL32 | 按 32 列一组的 tile 步长（如 33 列 2 行，ld ≥ 32×2 = 64） |
| COL4_4R2_8C | 32×8 复合 tile 步长（如 33 列 1 行，ld ≥ 32×8×1 = 256） |
| COL32_2R_4R4 | 32×32 复合 tile 步长（如 33 列 1 行，ld ≥ 32×32×1 = 1024） |

其中 rows / cols 指 `op` 变换后的逻辑行列；`op(A)`、`op(B)` 与 `C` 的逻辑维度必须相容。

**FP4 packed ld 语义**：FP4（`fp4x2`）为 2 元素/字节的 packed 布局，最小可寻址单位为「列对」（2 个相邻元素）。其 ld 以 **packed 字节步长**理解：逻辑 ld（元素数）经 `packedLd = (logicalLd + 1) / 2` 向上取整（ceil）换算为字节步长用于搬运；上表各 order 的 ld 下界（COL4_4R2_8C ≥ 32×8、COL32_2R_4R4 ≥ 32×32 元素）按 packed 字节换算（如 COL32_2R_4R4 单复合 tile 1024 元素 = 512 packed 字节）。因 packedLd 采用 ceil 向上取整，**FP4 的逻辑维度为奇数时合法支持**：奇数维补齐为整字节，最后一个字节的高 nibble 作为 padding 零填充，不返回错误。

## 精度标准

| dtype 路径 | 精度标准 | MARE 阈值 | MERE 阈值 |
|-----------|---------|-----------|-----------|
| INT8 / INT32 | 位精确 | MARE = 0 | MERE = 0 |
| FP32 | 单标杆 | MARE ≤ 10·2⁻¹³ | MERE ≤ 2⁻¹³ |
| FP16 | 单标杆 | MARE ≤ 10·2⁻¹⁰ | MERE ≤ 2⁻¹⁰ |
| BF16 | 单标杆 | MARE ≤ 10·2⁻⁷ | MERE ≤ 2⁻⁷ |
| FP8_E4M3FN | 单标杆（量化） | MARE ≤ 10·2⁻³ | MERE ≤ 2⁻³ |
| FP8_E5M2 | 单标杆（量化） | MARE ≤ 10·2⁻² | MERE ≤ 2⁻² |
| FP4_E2M1 | 单标杆（量化，按尾数位外推） | MARE ≤ 10·2⁻¹ | MERE ≤ 2⁻¹ |

整数路径为纯重排 / 整型缩放加法，要求与参考结果位精确一致（INT8 输出在 INT32 域计算后饱和钳位至 [-128, 127]）；浮点路径在 scaleType=FP32 域计算后转输出 dtype，按各 dtype 单标杆标准比对。FP8 路径在 scaleType=FP32 域计算后 `float→fp8`（RINT 舍入）写回，输出量化为主导误差，按量化标准比对（E4M3 阈值 2⁻³、E5M2 阈值 2⁻²）。FP4 路径在 BF16/FP32 域计算后 `bf16→fp4x2`（RINT 舍入）写回；E2M1 仅 1 位尾数、量化档位极少，阈值按尾数位外推（2⁻¹），输出量化误差为主导项。FP8/FP4 的参考结果须采用与设备 Cast 完全一致的舍入与饱和约定（量化档位错配会放大误差）。

## 接口

### 主接口（计算函数）

```c
aclblasStatus_t aclblasLtMatrixTransform(
    aclblasLtHandle_t lightHandle,
    aclblasLtMatrixTransformDesc_t transformDesc,
    const void* alpha,
    const void* A,
    aclblasLtMatrixLayout_t Adesc,
    const void* beta,
    const void* B,
    aclblasLtMatrixLayout_t Bdesc,
    void* C,
    aclblasLtMatrixLayout_t Cdesc,
    aclrtStream stream);
```

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| lightHandle | Host | 输入 | aclblasLtHandle_t | aclBLASLt 库上下文句柄，须经 `aclblasLtCreate` 创建 |
| transformDesc | Host | 输入 | aclblasLtMatrixTransformDesc_t | 变换描述符，承载 scaleType / pointerMode / TRANSA / TRANSB |
| alpha | Host | 输入 | const void* | A 的缩放系数，按 scaleType 解释（浮点/FP8 路径为 float，整数路径为 int32，FP4 路径为 bfloat16_t） |
| A | Device | 输入 | const void* | 输入矩阵 A，dtype/order/ld 由 Adesc 描述 |
| Adesc | Host | 输入 | aclblasLtMatrixLayout_t | A 的 layout 描述符 |
| beta | Host | 输入 | const void* | B 的缩放系数，按 scaleType 解释；beta=0 时 B/Bdesc 可为 NULL |
| B | Device | 输入 | const void* | 输入矩阵 B，可为 NULL（当 beta=0） |
| Bdesc | Host | 输入 | aclblasLtMatrixLayout_t | B 的 layout 描述符，可为 NULL（当 beta=0） |
| C | Device | 输出 | void* | 输出矩阵，dtype/order/ld 由 Cdesc 描述 |
| Cdesc | Host | 输入 | aclblasLtMatrixLayout_t | C 的 layout 描述符 |
| stream | Host | 输入 | aclrtStream | 执行流 |

**返回值**：

| 返回值 | 触发条件 |
|--------|---------|
| ACLBLAS_STATUS_SUCCESS | 执行成功（含空矩阵 no-op） |
| ACLBLAS_STATUS_NOT_INITIALIZED | lightHandle 为 NULL |
| ACLBLAS_STATUS_INVALID_VALUE | 描述符为 NULL、alpha/beta 为 NULL、参数冲突、逻辑维度不相容、ld 不满足下界等 |
| ACLBLAS_STATUS_NOT_SUPPORTED | dtype / order / op 组合不支持，或 batchCount > 1 |

### 变换描述符接口

```c
aclblasStatus_t aclblasLtMatrixTransformDescCreate(
    aclblasLtMatrixTransformDesc_t* transformDesc,
    aclDataType scaleType);

aclblasStatus_t aclblasLtMatrixTransformDescDestroy(
    const aclblasLtMatrixTransformDesc_t transformDesc);

aclblasStatus_t aclblasLtMatrixTransformDescSetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc,
    aclblasLtMatrixTransformDescAttribute_t attr,
    const void* buf,
    size_t sizeInBytes);

aclblasStatus_t aclblasLtMatrixTransformDescGetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc,
    aclblasLtMatrixTransformDescAttribute_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);
```

- `Create`：创建变换描述符并设置计算精度类型 `scaleType`（浮点路径与 FP8 路径传 `ACL_FLOAT`，整数路径传 `ACL_INT32`，FP4 路径传 `ACL_BF16`）。scaleType 须与 dtype 路径匹配，否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
- `Destroy`：销毁描述符。
- `SetAttribute` / `GetAttribute`：设置 / 查询变换属性。

**变换描述符属性 `aclblasLtMatrixTransformDescAttribute_t`**：

| 属性 | 枚举值 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE | 0 | int32_t (aclDataType) | Create 入参 | 计算精度类型，Create 时设置 |
| ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE | 1 | int32_t | HOST | alpha/beta 指针模式 |
| ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA | 2 | int32_t (aclblasOperation_t) | ACLBLAS_OP_N | 对 A 的变换 |
| ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB | 3 | int32_t (aclblasOperation_t) | ACLBLAS_OP_N | 对 B 的变换 |

### layout 描述符属性（复用 `aclblasLtMatrixLayoutAttribute_t`）

通过 `aclblasLtMatrixLayoutCreate` 创建后，用 `aclblasLtMatrixLayoutSetAttribute` 设置：

| 属性 | 枚举值 | 类型 | 说明 |
|------|--------|------|------|
| ACLBLASLT_MATRIX_LAYOUT_TYPE | 2 | uint32_t (aclDataType) | 矩阵数据类型 |
| ACLBLASLT_MATRIX_LAYOUT_ORDER | 3 | int32_t (aclblasLtOrder_t) | 内存布局 order，默认 COL |
| ACLBLASLT_MATRIX_LAYOUT_ROWS | 4 | uint64_t | 行数 |
| ACLBLASLT_MATRIX_LAYOUT_COLS | 5 | uint64_t | 列数 |
| ACLBLASLT_MATRIX_LAYOUT_LD | 6 | int64_t | leading dimension（元素数） |
| ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT | 0 | int32_t | batch 数，默认 1（首批仅支持 1） |

### order 枚举扩展 `aclblasLtOrder_t`

```c
typedef enum aclblasLtOrder {
    ACLBLASLT_ORDER_COL          = 0,  // 列主序
    ACLBLASLT_ORDER_ROW          = 1,  // 行主序
    ACLBLASLT_ORDER_COL32        = 2,  // 32 列复合 tile，组内列主序
    ACLBLASLT_ORDER_COL4_4R2_8C  = 3,  // 32 列 × 8 行复合量化布局
    ACLBLASLT_ORDER_COL32_2R_4R4 = 4,  // 32 列 × 32 行复合量化布局
} aclblasLtOrder_t;
```

> 其中 `COL32` / `COL4_4R2_8C` / `COL32_2R_4R4` 为本算子新增追加的枚举值（仅追加，不改动 COL/ROW 旧值，保持 ABI 兼容）。

## 调用示例

以下示例展示 FP32 列主序、`C = 2.0·A + 1.0·B`（16×16）的完整调用流程：描述符创建 → 设属性 → 调用 → 销毁。

```cpp
#include "acl/acl.h"
#include "cann_ops_blasLt.h"

void RunMatrixTransform(aclblasLtHandle_t handle, aclrtStream stream,
                        void* dA, void* dB, void* dC) {
    const uint64_t rows = 16, cols = 16;
    const int64_t  ld   = 16;            // 列主序 ld >= rows
    float alpha = 2.0f, beta = 1.0f;

    // 1) 创建变换描述符，scaleType = FP32（浮点路径）
    aclblasLtMatrixTransformDesc_t transformDesc = nullptr;
    aclblasLtMatrixTransformDescCreate(&transformDesc, ACL_FLOAT);

    // 2) 设置变换属性（op = N，可省略，默认即 ACLBLAS_OP_N）
    int32_t opN = ACLBLAS_OP_N;
    aclblasLtMatrixTransformDescSetAttribute(
        transformDesc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opN, sizeof(int32_t));
    aclblasLtMatrixTransformDescSetAttribute(
        transformDesc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opN, sizeof(int32_t));

    // 3) 创建 A / B / C 的 layout 描述符，并设置 order = COL
    aclblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    int32_t order = ACLBLASLT_ORDER_COL;
    aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, rows, cols, ld);
    aclblasLtMatrixLayoutSetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, rows, cols, ld);
    aclblasLtMatrixLayoutSetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT, rows, cols, ld);
    aclblasLtMatrixLayoutSetAttribute(Cdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));

    // 4) 调用主接口：C = alpha * A + beta * B
    aclblasLtMatrixTransform(
        handle, transformDesc,
        &alpha, dA, Adesc,
        &beta,  dB, Bdesc,
        dC, Cdesc, stream);
    aclrtSynchronizeStream(stream);

    // 5) 销毁描述符
    aclblasLtMatrixTransformDescDestroy(transformDesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
}
```

**纯转换 / 单输入用法**：令 `alpha = 1.0`、`beta = 0.0`，并将 `B` 与 `Bdesc` 传入 `nullptr`，即可完成 `C = op(A)` 的纯布局/类型转换：

```cpp
float alpha = 1.0f, beta = 0.0f;
aclblasLtMatrixTransform(
    handle, transformDesc,
    &alpha, dA, Adesc,
    &beta,  /*B=*/nullptr, /*Bdesc=*/nullptr,
    dC, Cdesc, stream);
```

**整数路径用法**：变换描述符 `Create` 时传 `ACL_INT32` 作为 scaleType，layout 的 dtype 设为 `ACL_INT8` 或 `ACL_INT32`，`alpha` / `beta` 以 `int32_t` 指针传入。复杂量化布局（COL4_4R2_8C / COL32_2R_4R4）支持 INT8 / INT32 / FP8 / FP4 量化路径 dtype。

**FP8 路径用法**：变换描述符 `Create` 时传 `ACL_FLOAT` 作为 scaleType，layout 的 dtype 设为 `ACL_FLOAT8_E4M3FN` 或 `ACL_FLOAT8_E5M2`，`alpha` / `beta` 以 `float` 指针传入。FP8 支持全部 5 种 order（含复杂量化布局）与全部 op（N/T，C 等价 T），可与 FP32/FP16/BF16 经 float 跨 dtype 转换：

```cpp
// FP8 E4M3FN 列主序：C = 1.0 * A（纯转换，单输入），16×16
float alpha = 1.0f, beta = 0.0f;
aclblasLtMatrixTransformDesc_t transformDesc = nullptr;
aclblasLtMatrixTransformDescCreate(&transformDesc, ACL_FLOAT);   // FP8 路径 scaleType = FP32

aclblasLtMatrixLayout_t Adesc = nullptr, Cdesc = nullptr;
int32_t order = ACLBLASLT_ORDER_COL;
aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT8_E4M3FN, 16, 16, 16);
aclblasLtMatrixLayoutSetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT8_E4M3FN, 16, 16, 16);
aclblasLtMatrixLayoutSetAttribute(Cdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));

aclblasLtMatrixTransform(handle, transformDesc,
    &alpha, dA, Adesc,
    &beta,  /*B=*/nullptr, /*Bdesc=*/nullptr,
    dC, Cdesc, stream);
```

**FP4 路径用法**：变换描述符 `Create` 时传 `ACL_BF16` 作为 scaleType，layout 的 dtype 设为 `ACL_FLOAT4_E2M1`，`alpha` / `beta` 以 `bfloat16_t` 指针传入。FP4 为 `fp4x2` packed（2 元素/字节），ld 按 packed 字节理解，由 `packedLd = (logicalLd + 1) / 2` 向上取整换算；逻辑维度为奇数时合法支持（ceil 补齐，尾字节高 nibble 零填充）。FP4 支持全部 5 种 order（含复杂量化布局，复杂布局走 bf16 解包/重打包路径）与全部 op，可经 bf16 与 BF16/FP32 跨 dtype 转换。

## 编译

```bash
bash build.sh --ops=blasLtMatrixTransform --soc=ascend950
```

## 测试

运行精度 ST（GTest + CSV 参数化）：

```bash
bash build.sh --ops=blasLtMatrixTransform --soc=ascend950 --run
```

精度对比通过时输出：

```bash
[Success] Case accuracy is verification passed.
```

测试工程位于 `test/blasLtMatrixTransform/`，用例由 `arch35/blasLtMatrixTransform_test.csv` 驱动，覆盖各 dtype（含 FP8 E4M3FN/E5M2、FP4 E2M1）、order（含复杂量化布局）、op 与边界场景。FP8/FP4 用例的参考结果采用与设备 Cast 一致的量化舍入语义。
