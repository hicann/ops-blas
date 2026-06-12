# aclblasSsymm

对称矩阵与普通矩阵的乘法运算。

## 功能描述

实现对称矩阵与普通矩阵的乘法，数学定义：
- LEFT 模式：`C := alpha * A * B + beta * C`
- RIGHT 模式：`C := alpha * B * A + beta * C`

其中 A 为对称矩阵（仅存储上三角或下三角），B 和 C 为普通矩阵。

## 接口定义

```cpp
aclblasStatus_t aclblasSsymm(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m,
    int64_t n,
    const float *alpha,
    const float *A,
    int64_t lda,
    const float *B,
    int64_t ldb,
    const float *beta,
    float *C,
    int64_t ldc
);
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|----------|------|
| handle | 输入 | ACL-BLAS 句柄 |
| side | 输入 | A 矩阵位置（ACLBLAS_SIDE_LEFT / ACLBLAS_SIDE_RIGHT） |
| uplo | 输入 | A 矩阵存储模式（ACLBLAS_LOWER / ACLBLAS_UPPER） |
| m | 输入 | C 矩阵行数 |
| n | 输入 | C 矩阵列数 |
| alpha | 输入 | 标量系数 α（Host 侧指针） |
| A | 输入 | 对称矩阵（side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 侧） |
| lda | 输入 | A 矩阵 leading dimension |
| B | 输入 | m×n 矩阵（Device 侧） |
| ldb | 输入 | B 矩阵 leading dimension |
| beta | 输入 | 标量系数 β（Host 侧指针） |
| C | 输入/输出 | m×n 矩阵（输入旧值，输出新值，Device 侧） |
| ldc | 输入 | C 矩阵 leading dimension |

## 约束条件

### 参数约束
- m, n ≥ 0
- lda ≥ max(1, aDim)，其中 aDim = (side=LEFT ? m : n)
- ldb ≥ max(1, n)
- ldc ≥ max(1, n)
- m, n, lda, ldb, ldc ≤ UINT32_MAX
- 所有指针参数不能为 null（alpha, beta, A, B, C）
- 缓冲区大小：rows×cols×sizeof(float) 不能溢出 size_t

### 快速返回
- 当 m = 0 或 n = 0 时，直接返回 ACLBLAS_STATUS_SUCCESS，不执行任何计算

### 数据类型
- 支持：FP32（单精度浮点数）

### 支持的芯片架构

- ✅ **arch22**: Ascend910B, Ascend910_93 (A2/A3 训练推理系列)
- ❌ **arch35**: Ascend950 (A5 训练推理系列) - 暂不支持

### 精度要求
- 精度标准：浮点计算类社区标准（单标杆）
- MARE（最大绝对相对误差）：≤ 10 * 2^-13
- MERE（最大均方根相对误差）：≤ 2^-13

## 使用示例

```cpp
#include "cann_ops_blas.h"

// 示例：LEFT + LOWER 模式，256×256 矩阵
aclblasHandle handle;
aclblasCreate(&handle);

const int m = 256, n = 256;
float alpha = 1.25f, beta = 0.5f;

// 分配并初始化矩阵（A 为 m×m 对称矩阵，B 和 C 为 m×n 矩阵）
float *A, *B, *C;  // Device 侧指针，已通过 aclrtMalloc 分配

aclblasStatus_t status = aclblasSsymm(
    handle,
    ACLBLAS_SIDE_LEFT,    // A 在左侧
    ACLBLAS_LOWER,        // A 存储下三角
    m, n,
    &alpha,
    A, m,                 // lda = m
    B, n,                 // ldb = n
    &beta,
    C, n                  // ldc = n
);

if (status != ACLBLAS_STATUS_SUCCESS) {
    // 错误处理
}

aclblasDestroy(handle);
```

## 性能数据

以下性能数据在 Ascend910B（arch22, CANN 9.0.0）上采集，设备满频 1800 MHz。

### LEFT 路径性能

| Shape | Side | Uplo | 耗时 (μs) | Block Dim | 主导单元 | 说明 |
|-------|------|------|----------|-----------|---------|------|
| 256×256 | LEFT | LOWER | 7193 | 8 (vector), 1 (cube) | Vector (>90%) | 5 阶段流水线 |
| 512×512 | LEFT | LOWER | 12426 | 8 (vector), 1 (cube) | Vector (>90%) | clear_partial 调用 2 次 |
| 1024×1024 | LEFT | LOWER | 23113 | 8 (vector), 1 (cube) | Vector (>90%) | - |

### RIGHT 路径性能

| Shape | Side | Uplo | 耗时 (μs) | Block Dim | 主导单元 | 说明 |
|-------|------|------|----------|-----------|---------|------|
| 256×256 | RIGHT | LOWER | 1606 | 8 (vector), 1 (cube) | Vector (~75%) | 3 阶段流水线 |
| 512×512 | RIGHT | LOWER | 2969 | 8 (vector), 1 (cube) | Vector (~75%) | - |
| 1024×1024 | RIGHT | LOWER | >55000 | 8 (vector) | Vector | ⚠️ 路径切换导致性能退化 |

### 性能特征

- **RIGHT 路径优势**：小中规模下（256×256, 512×512），RIGHT 路径性能优于 LEFT 路径约 4-5 倍
- **核间并行**：使用多核并行（Block Dim=8），充分利用向量计算核心
- **流水线**：LEFT 路径为 5 阶段流水线（clear_partial → pack → dense → accum → postprocess），RIGHT 路径为 3 阶段流水线（pack → dense → accum）
- **Cube 利用率**：LEFT 路径 Cube 计算占比 <10%，RIGHT 路径约 15-20%

## 已知限制

- 1024×1024 RIGHT 路径在当前版本存在性能退化（路径切换导致），后续版本将优化
- Vector 计算单元利用率 <20%（受 ssymm 算法的数据依赖特性限制）
- 内存带宽利用率 <80%（算子为 compute bound）

## 返回值

| 返回值 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 成功 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空 |
| ACLBLAS_STATUS_INVALID_ENUM | side 或 uplo 枚举值非法 |
| ACLBLAS_STATUS_INVALID_VALUE | 参数不满足约束条件（负数维度、leading dimension 过小、超出 uint32 范围、缓冲区溢出、空指针等） |
| ACLBLAS_STATUS_EXECUTION_FAILED | Kernel 执行失败 |

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2026-06-03 | 二轮优化：清理诊断代码、动态化核数、修复编码规范问题 |
| v1.0 | - | 初始版本：实现 LEFT/RIGHT 路径、三路 backend 分派 |

## 参考资料

- [BLAS SSYMM 标准](http://www.netlib.org/blas/#_ssymm) - 数学定义和参数规范
- CANN API 文档 - `aclrtGetDeviceInfo` 动态获取设备核数
- ops-blas 编码规范 - R1-R4 编码约束

## 调试环境变量

- `SSYMM_DEBUG_PLAN=1` - 打印 backend 选择信息（side/uplo/backend）
