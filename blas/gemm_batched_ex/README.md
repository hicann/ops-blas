# aclblasGemmBatchedEx

## 接口

```c
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* const Aarray[],
    aclDataType Atype,
    int lda,
    const void* const Barray[],
    aclDataType Btype,
    int ldb,
    const void* beta,
    void* const Carray[],
    aclDataType Ctype,
    int ldc,
    int batchCount,
    aclblasComputeType_t computeType,
    aclblasGemmAlgo_t algo);
```

### 功能说明

执行批量矩阵乘法运算：

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  for i ∈ [0, batchCount - 1]
```

其中 `op(X)` 根据转置标志决定：`N` 为不转置，`T` 为转置，`C` 为共轭转置。所有矩阵以列主序（Column-Major）存储。所有批次共享相同的维度 (m, n, k)、leading dimensions (lda, ldb, ldc) 和转置标志 (transa, transb)，每个批次的矩阵指针通过设备侧指针数组独立指定。

### 参数说明

| 参数 | 位置 | 方向 | 类型 | 说明 |
|------|------|------|------|------|
| handle | Host | 输入 | aclblasHandle_t | 库上下文句柄 |
| transa | Host | 输入 | aclblasOperation_t | 矩阵 A 的转置操作（N / T / C） |
| transb | Host | 输入 | aclblasOperation_t | 矩阵 B 的转置操作（N / T / C） |
| m | Host | 输入 | int | op(A[i]) 的行数，也是 C[i] 的行数，须 ≥ 0 |
| n | Host | 输入 | int | op(B[i]) 的列数，也是 C[i] 的列数，须 ≥ 0 |
| k | Host | 输入 | int | op(A[i]) 的列数，也是 op(B[i]) 的行数，须 ≥ 0 |
| alpha | Host | 输入 | const void\* | 标量缩放因子，类型由 computeType 决定 |
| Aarray | Device | 输入 | const void\* const [] | 设备侧指针数组，每个元素指向 Atype 类型的矩阵 A[i] |
| Atype | Host | 输入 | aclDataType | 矩阵 A 的数据类型 |
| lda | Host | 输入 | int | 矩阵 A[i] 的 leading dimension |
| Barray | Device | 输入 | const void\* const [] | 设备侧指针数组，每个元素指向 Btype 类型的矩阵 B[i] |
| Btype | Host | 输入 | aclDataType | 矩阵 B 的数据类型 |
| ldb | Host | 输入 | int | 矩阵 B[i] 的 leading dimension |
| beta | Host | 输入 | const void\* | 标量缩放因子，类型由 computeType 决定 |
| Carray | Device | 输入/输出 | void\* const [] | 设备侧指针数组，每个元素指向 Ctype 类型的矩阵 C[i] |
| Ctype | Host | 输入 | aclDataType | 矩阵 C 的数据类型 |
| ldc | Host | 输入 | int | 矩阵 C[i] 的 leading dimension |
| batchCount | Host | 输入 | int | 批次数量，须 ≥ 0 |
| computeType | Host | 输入 | aclblasComputeType_t | 计算精度类型 |
| algo | Host | 输入 | aclblasGemmAlgo_t | GEMM 算法选择（当前仅支持 ACLBLAS_GEMM_DEFAULT） |

### Leading Dimension 约束

| 条件 | 约束 |
|------|------|
| transa == N | lda ≥ max(1, m) |
| transa == T 或 C | lda ≥ max(1, k) |
| transb == N | ldb ≥ max(1, k) |
| transb == T 或 C | ldb ≥ max(1, n) |
| 任意 | ldc ≥ max(1, m) |

## 支持规格

| 项目 | 内容 |
|------|------|
| 目标芯片 | Ascend950 (PR/DT) |
| 目标架构 | arch35 (DAV_3510) |

### 支持的数据类型组合

| 序号 | Atype | Btype | Ctype | computeType | alpha/beta 宿主类型 | 说明 |
|------|-------|-------|-------|-------------|-------------------|------|
| 1 | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACLBLAS_COMPUTE_16F | half (FP16) | FP16 纯精度计算 |
| 2 | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP16 输入，FP32 累加 |
| 3 | ACL_BF16 | ACL_BF16 | ACL_BF16 | ACLBLAS_COMPUTE_32F | float (FP32) | BF16 输入，FP32 累加 |
| 4 | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 E4M3 输入，FP16 输出 |
| 5 | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 E5M2 输入，FP16 输出 |
| 6 | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 混合输入 (E4M3×E5M2) |
| 7 | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 混合输入 (E5M2×E4M3) |

不在上表中的 Atype/Btype/Ctype/computeType 组合将返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

## 已知限制

1. **FP8 输入必须使用 COMPUTE_32F**：当 Atype 或 Btype 为 FP8 类型（ACL_FLOAT8_E4M3FN / ACL_FLOAT8_E5M2）时，computeType 必须为 ACLBLAS_COMPUTE_32F，否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

2. **FP8 精度受限**：FP8 数据类型本身的表示精度有限（E4M3 尾数 3 位，E5M2 尾数 2 位），输出精度低于 FP16/BF16 类型。FP8 E4M3 的 MERE 阈值约为 2^-3，FP8 E5M2 的 MERE 阈值约为 2^-2。

3. **临时缓冲上限**：当 alpha ≠ 1 或 beta ≠ 0 时需要分配临时缓冲（大小 = m × n × batchCount × elemSize）。临时缓冲设有 256 MB 上限，超出时自动采用分批处理策略，可能引入额外的 kernel launch 开销。

4. **算法选择**：当前仅支持 `ACLBLAS_GEMM_DEFAULT`，传入其他算法将返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

## 编译

```bash
bash build.sh --ops=gemm_batched_ex
```

## 测试

```bash
bash build.sh --ops=gemm_batched_ex --run
```

## 调用示例

以下示例演示列主序（Column-Major）下的批量矩阵乘法调用。计算 `C[i] = 1.0 * A[i] * B[i] + 0.0 * C[i]`，其中 A 为 M×K，B 为 K×N，C 为 M×N，共 batchCount 个批次。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 矩阵维度（列主序）
int M = 256;
int N = 256;
int K = 128;
int batchCount = 4;

// 初始化 ACL 和 BLAS handle
aclInit(nullptr);
int32_t deviceId = 0;
aclrtSetDevice(deviceId);
aclblasHandle_t handle;
aclblasCreate(&handle);

// 在设备侧分配矩阵内存（列主序存储）
size_t sizeA = (size_t)M * K * sizeof(half);
size_t sizeB = (size_t)K * N * sizeof(half);
size_t sizeC = (size_t)M * N * sizeof(half);

void* dA[batchCount];
void* dB[batchCount];
void* dC[batchCount];
for (int i = 0; i < batchCount; i++) {
    aclrtMalloc(&dA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC[i], sizeC, ACL_MEM_MALLOC_HUGE_FIRST);
    // ... 填充数据 ...
}

// 构造设备侧指针数组（在设备内存中）
void** dAarray;
void** dBarray;
void** dCarray;
aclrtMalloc((void**)&dAarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dBarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dCarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dAarray, batchCount * sizeof(void*), dA, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dBarray, batchCount * sizeof(void*), dB, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dCarray, batchCount * sizeof(void*), dC, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);

// 设置 alpha 和 beta（COMPUTE_32F 时为 float 类型）
float alpha = 1.0f;
float beta  = 0.0f;

// 调用 aclblasGemmBatchedEx
// 列主序：A 为 M×K（lda=M），B 为 K×N（ldb=K），C 为 M×N（ldc=M）
aclblasStatus_t status = aclblasGemmBatchedEx(
    handle,
    ACLBLAS_OP_N,                          // transa: 不转置
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,                               // 矩阵维度
    &alpha,                                // alpha (float*)
    (const void* const*)dAarray,           // Aarray (设备侧指针数组)
    ACL_FLOAT16,                           // Atype
    M,                                     // lda
    (const void* const*)dBarray,           // Barray (设备侧指针数组)
    ACL_FLOAT16,                           // Btype
    K,                                     // ldb
    &beta,                                 // beta (float*)
    (void* const*)dCarray,                 // Carray (设备侧指针数组)
    ACL_FLOAT16,                           // Ctype
    M,                                     // ldc
    batchCount,                            // 批次数量
    ACLBLAS_COMPUTE_32F,                   // FP32 累加
    ACLBLAS_GEMM_DEFAULT                   // 默认算法
);

// 同步等待完成
aclrtSynchronizeStream(nullptr);

// 清理资源
aclblasDestroy(handle);
for (int i = 0; i < batchCount; i++) {
    aclrtFree(dA[i]);
    aclrtFree(dB[i]);
    aclrtFree(dC[i]);
}
aclrtFree(dAarray);
aclrtFree(dBarray);
aclrtFree(dCarray);
aclrtResetDevice(deviceId);
aclFinalize();
```

### 转置调用示例

当需要对 A 进行转置时（`transa = ACLBLAS_OP_T`），A 的物理存储为 K×M（列主序），lda ≥ max(1, K)：

```cpp
// A^T * B: transa=T, A 物理存储为 K×M (lda=K)
aclblasStatus_t status = aclblasGemmBatchedEx(
    handle,
    ACLBLAS_OP_T,                          // transa: 转置 A
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,
    &alpha,
    (const void* const*)dAarray, ACL_FLOAT16, K,   // lda = K（转置时）
    (const void* const*)dBarray, ACL_FLOAT16, K,
    &beta,
    (void* const*)dCarray, ACL_FLOAT16, M,
    batchCount,
    ACLBLAS_COMPUTE_32F,
    ACLBLAS_GEMM_DEFAULT
);
```
