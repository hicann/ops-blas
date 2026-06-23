# aclblasGemmEx

通用矩阵乘法扩展接口（GEMM Ex），支持 A、B、C 矩阵使用独立数据类型。

```
C = α * op(A) * op(B) + β * C
```

其中 `op(X)` 根据转置参数决定为 `X`、`X^T` 或 `X^H`。矩阵采用列主序（Column-Major）存储。

## 接口

```c
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle,
    aclblasOperation_t transA,
    aclblasOperation_t transB,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* A,
    aclDataType Atype,
    int lda,
    const void* B,
    aclDataType Btype,
    int ldb,
    const void* beta,
    void* C,
    aclDataType Ctype,
    int ldc,
    aclblasComputeType_t computeType,
    aclblasGemmAlgo_t algo);
```

### 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| handle | Host | 输入 | aclblasHandle_t | ops-blas 库上下文句柄 |
| transA | Host | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：`ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置） |
| transB | Host | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transA） |
| m | Host | 输入 | int | op(A) 和 C 的行数 |
| n | Host | 输入 | int | op(B) 和 C 的列数 |
| k | Host | 输入 | int | op(A) 的列数和 op(B) 的行数 |
| alpha | Host | 输入 | const void* | 标量 α，指向 float 类型的指针 |
| A | Device | 输入 | const void* | 矩阵 A 的设备内存指针 |
| Atype | Host | 输入 | aclDataType | 矩阵 A 的数据类型 |
| lda | Host | 输入 | int | 矩阵 A 的主维度（列主序） |
| B | Device | 输入 | const void* | 矩阵 B 的设备内存指针 |
| Btype | Host | 输入 | aclDataType | 矩阵 B 的数据类型 |
| ldb | Host | 输入 | int | 矩阵 B 的主维度（列主序） |
| beta | Host | 输入 | const void* | 标量 β，指向 float 类型的指针 |
| C | Device | 输入/输出 | void* | 矩阵 C 的设备内存指针 |
| Ctype | Host | 输入 | aclDataType | 矩阵 C 的数据类型 |
| ldc | Host | 输入 | int | 矩阵 C 的主维度（列主序），`ldc >= max(1, m)` |
| computeType | Host | 输入 | aclblasComputeType_t | 计算精度类型 |
| algo | Host | 输入 | aclblasGemmAlgo_t | 算法选择，当前仅支持 `ACLBLAS_GEMM_DEFAULT` |

### 主维度约束（列主序）

| 条件 | lda 约束 | ldb 约束 |
|------|---------|---------|
| transA = N | `lda >= max(1, m)` | — |
| transA = T/C | `lda >= max(1, k)` | — |
| transB = N | — | `ldb >= max(1, k)` |
| transB = T/C | — | `ldb >= max(1, n)` |

### 返回值

| 返回值 | 说明 |
|--------|------|
| `ACLBLAS_STATUS_SUCCESS` | 执行成功 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空指针 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数值非法（维度为负、主维度不足、指针为空等） |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 数据类型组合不支持或算法选项不支持 |
| `ACLBLAS_STATUS_EXECUTION_FAILED` | 执行失败 |

## 支持规格

| 项目 | 内容 |
|------|------|
| 目标芯片 | Ascend950PR / Ascend950DT |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMD membase |

### 支持的数据类型组合

| 序号 | computeType | Atype | Btype | Ctype | 说明 |
|-----|------------|-------|-------|-------|------|
| 1 | ACLBLAS_COMPUTE_16F | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | FP16 纯精度，最高性能 |
| 2 | ACLBLAS_COMPUTE_32F | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | FP16 输入 + FP32 累加 |
| 3 | ACLBLAS_COMPUTE_32F | ACL_BF16 | ACL_BF16 | ACL_BF16 | BF16 输入 + FP32 累加 |
| 5 | ACLBLAS_COMPUTE_32F | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | FP8 E4M3 输入 → FP16 输出 |
| 6 | ACLBLAS_COMPUTE_32F | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | FP8 E5M2 输入 → FP16 输出 |
| 7 | ACLBLAS_COMPUTE_32F | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | FP8 混合输入（E4M3×E5M2） |
| 8 | ACLBLAS_COMPUTE_32F | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | FP8 混合输入（E5M2×E4M3） |

### 精度标准

| 输出数据类型 | MERE Threshold | MARE Threshold |
|------------|---------------|---------------|
| FP16 | < 2^-10 (~0.000977) | < 10 × 2^-10 (~0.00977) |
| BF16 | < 2^-7 (~0.00781) | < 10 × 2^-7 (~0.0781) |
| FP8 E4M3 | < 2^-3 (~0.125) | < 10 × 2^-3 (~1.25) |
| FP8 E5M2 | < 2^-2 (~0.25) | < 10 × 2^-2 (~2.5) |

- **MERE**：平均相对误差（Mean Relative Error）
- **MARE**：最大相对误差（Maximum Relative Error）

## 编译

```bash
bash build.sh --ops=gemm
```

## 测试

```bash
bash build.sh --ops=gemm --run
```

## 调用示例

### FP16 矩阵乘法（列主序）

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 矩阵维度: C(128x256) = A(128x64) * B(64x256)
int m = 128, n = 256, k = 64;
float alpha = 1.0f, beta = 0.0f;

// 分配设备内存 (列主序)
size_t sizeA = m * k * sizeof(uint16_t);  // FP16
size_t sizeB = k * n * sizeof(uint16_t);
size_t sizeC = m * n * sizeof(uint16_t);

void *devA, *devB, *devC;
aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST);

// ... 初始化 A, B 数据 (通过 aclrtMemcpy H2D) ...

// 创建 handle
aclblasHandle_t handle;
aclblasCreate(&handle);

// 调用 aclblasGemmEx
aclblasStatus_t status = aclblasGemmEx(
    handle,
    ACLBLAS_OP_N, ACLBLAS_OP_N,     // transA, transB
    m, n, k,                         // 矩阵维度
    &alpha,                          // α
    devA, ACL_FLOAT16, m,            // A, Atype, lda
    devB, ACL_FLOAT16, k,            // B, Btype, ldb
    &beta,                           // β
    devC, ACL_FLOAT16, m,            // C, Ctype, ldc
    ACLBLAS_COMPUTE_16F,             // computeType
    ACLBLAS_GEMM_DEFAULT);           // algo

// 清理
aclblasDestroy(handle);
aclrtFree(devA);
aclrtFree(devB);
aclrtFree(devC);
```

### FP8 混合精度矩阵乘法

```cpp
// FP8 E4M3FN 输入 → FP16 输出，FP32 累加
int m = 256, n = 256, k = 128;
float alpha = 1.0f, beta = 0.0f;

size_t sizeA = m * k * sizeof(uint8_t);   // FP8 = 1 byte
size_t sizeB = k * n * sizeof(uint8_t);
size_t sizeC = m * n * sizeof(uint16_t);  // FP16 output

void *devA, *devB, *devC;
aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST);

// ... 初始化数据 ...

aclblasHandle_t handle;
aclblasCreate(&handle);

aclblasStatus_t status = aclblasGemmEx(
    handle,
    ACLBLAS_OP_N, ACLBLAS_OP_N,
    m, n, k,
    &alpha,
    devA, ACL_FLOAT8_E4M3FN, m,      // FP8 E4M3FN 输入
    devB, ACL_FLOAT8_E4M3FN, k,      // FP8 E4M3FN 输入
    &beta,
    devC, ACL_FLOAT16, m,             // FP16 输出
    ACLBLAS_COMPUTE_32F,              // FP32 累加（FP8 必须使用 FP32 computeType）
    ACLBLAS_GEMM_DEFAULT);

aclblasDestroy(handle);
aclrtFree(devA);
aclrtFree(devB);
aclrtFree(devC);
```

### 带转置和 alpha/beta 的矩阵乘法

```cpp
// C = 2.0 * A^T * B + 0.5 * C
// A 原始为 k×m (列主序)，转置后为 m×k
// B 为 k×n，C 为 m×n
int m = 64, n = 128, k = 256;
float alpha = 2.0f, beta = 0.5f;

// A 存储为 k×m 列主序，lda = k（因为 transA = T）
// B 存储为 k×n 列主序，ldb = k
// C 存储为 m×n 列主序，ldc = m
size_t sizeA = k * m * sizeof(uint16_t);
size_t sizeB = k * n * sizeof(uint16_t);
size_t sizeC = m * n * sizeof(uint16_t);

void *devA, *devB, *devC;
aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST);

// ... 初始化数据（C 需初始化，因 beta != 0）...

aclblasHandle_t handle;
aclblasCreate(&handle);

aclblasStatus_t status = aclblasGemmEx(
    handle,
    ACLBLAS_OP_T, ACLBLAS_OP_N,      // A 转置，B 不转置
    m, n, k,
    &alpha,
    devA, ACL_FLOAT16, k,             // lda = k (transA=T 时 lda >= max(1,k))
    devB, ACL_FLOAT16, k,             // ldb = k (transB=N 时 ldb >= max(1,k))
    &beta,
    devC, ACL_FLOAT16, m,             // ldc = m
    ACLBLAS_COMPUTE_32F,              // FP32 累加提高精度
    ACLBLAS_GEMM_DEFAULT);

aclblasDestroy(handle);
aclrtFree(devA);
aclrtFree(devB);
aclrtFree(devC);
```

## 已知限制

1. **算法选择**：当前仅支持 `ACLBLAS_GEMM_DEFAULT`（默认算法），其他显式算法选项（`ACLBLAS_GEMM_ALGO0` ~ `ACLBLAS_GEMM_ALGO7`）暂不支持，传入将返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

2. **FP8 约束**：
   - FP8 输入必须使用 `ACLBLAS_COMPUTE_32F` 作为 computeType。
   - FP8 不支持作为输出类型（Ctype），输出需为 FP16/BF16/FP32。
   - FP8 E4M3FN 格式不支持 NaN/Inf 编码，超出动态范围（±448）的值将被硬件自动饱和至最大值。

4. **alpha/beta 精度**：alpha 和 beta 参数始终以 FP32（float）类型传入。当 Ctype 为 FP16/BF16 且 `alpha != 1.0` 或 `beta != 0.0` 时，内部采用 FP32 中间输出路径以保证 alpha/beta 缩放精度，避免中间量化误差。

5. **性能 Scalar Bound**：alpha/beta 后处理在 Host 侧执行（涉及 Device↔Host 数据搬运），当 `alpha != 1.0` 或 `beta != 0.0` 时会引入额外的数据搬运开销。对于 `alpha = 1.0, beta = 0.0` 的场景，算子可直接输出最终结果，性能最优。

6. **主维度打包**：当 `lda > physRows`（或 ldb/ldc 类似情况）时，Host 侧会进行列主序 Pack/Unpack 操作以适配 Matmul API 的连续存储要求，会引入额外的 Host↔Device 数据搬运开销。
