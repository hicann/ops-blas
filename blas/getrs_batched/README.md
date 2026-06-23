# aclblasSgetrsBatched

## 接口

```c
aclblasStatus_t aclblasSgetrsBatched(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int n,
    int nrhs,
    const float *const Aarray[],
    int lda,
    const int *devIpiv,
    float *const Barray[],
    int ldb,
    int *info,
    int batchCount);
```

## 功能

对一批已经由 `aclblasSgetrfBatched` 完成 LU 分解的 n×n 方阵，批量求解线性方程组：

```
op(A[i]) * X[i] = B[i],   i = 0, 1, ..., batchCount - 1
```

| 符号 | 含义 |
|------|------|
| `A[i]` | 第 i 个 n×n 方阵（列主序存储），已由 `aclblasSgetrfBatched` 完成 LU 分解，存储 L（单位下三角，对角线为 1 不显式存储）和 U（上三角） |
| `op(A[i])` | 由 `trans` 参数决定：`ACLBLAS_OP_N` → A[i]；`ACLBLAS_OP_T` → A[i]^T；`ACLBLAS_OP_C` → A[i]^H（实数域等价于 A[i]^T） |
| `X[i]` | 第 i 个 n×nrhs 解矩阵，结果覆盖写入 `Barray[i]` |
| `B[i]` | 第 i 个 n×nrhs 右端矩阵（列主序存储），输入时为右端项，输出时被覆盖为解 X[i] |
| `P[i]` | 置换矩阵，由 `devIpiv` 中的主元序列构造（`devIpiv` 为 `NULL` 时 P = I） |

**算法流程**（基于 LAPACK sgetrs 标准）：

**当 trans == ACLBLAS_OP_N 时**（求解 A*X = B，即 P*L*U*X = B）：
1. 对 B[i] 应用置换 P[i]：B' = P*B（按主元序列交换行）
2. 前向求解 L*Y = B'（利用 L 的单位下三角结构逐列前向代入，原地覆盖 B）
3. 后向求解 U*X = Y（利用 U 的上三角结构逐列后向代入，原地覆盖 B）
4. 最终 B[i] = X[i]

**当 trans == ACLBLAS_OP_T / ACLBLAS_OP_C 时**（求解 A^T*X = B，即 U^T*L^T*P*X = B）：
1. 前向求解 U^T*Y = B（利用 U^T 的下三角结构逐列前向代入，原地覆盖 B）
2. 后向求解 L^T*Z = Y（利用 L^T 的上三角结构逐列后向代入，原地覆盖 B）
3. 对 B[i] 应用逆置换 P^T：X = P^T*Z（按主元序列逆序交换行）
4. 最终 B[i] = X[i]

**奇异矩阵行为**：若 `U[i](k,k) == 0`（U 精确奇异），求解结果未定义（可能包含 inf/nan），`info` 不上报奇异信息。调用方应在 `aclblasSgetrfBatched` 阶段检查 `infoArray` 确认矩阵非奇异。

> 接口对齐 LAPACK sgetrs 标准（[NETLIB 文档](http://www.netlib.org/lapack/single/sgetrs.f)）。

### 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| `handle` | Host | 输入 | `aclblasHandle_t` | ops-blas 库上下文句柄，内部携带 stream |
| `trans` | Host | 输入 | `aclblasOperation_t` | 转置操作类型：`ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置，实数域等价于转置） |
| `n` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的行数和列数（方阵边长），n >= 0 |
| `nrhs` | Host | 输入 | `int` | 每个矩阵 `Barray[i]` 的列数（右端项数量），nrhs >= 0，nrhs <= 256 |
| `Aarray` | Device | 输入 | `const float *const []` | **Device 侧指针数组**：每个元素 `Aarray[i]` 指向 Device 内存中已经 LU 分解的 n×n float 矩阵（列主序，leading dimension = `lda`） |
| `lda` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的 leading dimension，lda >= max(1, n) |
| `devIpiv` | Device | 输入 | `const int *` | 大小为 n × batchCount 的数组，存储每个矩阵的主元序列。对应 `aclblasSgetrfBatched` 的输出参数 `PivotArray`，即 `devIpiv` = `PivotArray`。`devIpiv[i * n + j]` 为第 i 个 batch 第 j 步的行交换索引（1-indexed，LAPACK 约定）。**可为 NULL**，表示 LU 分解未使用主元选取（P = I） |
| `Barray` | Device | 输入/输出 | `float *const []` | **Device 侧指针数组**：每个元素 `Barray[i]` 指向 Device 内存中 n×nrhs float 矩阵（列主序，leading dimension = `ldb`）。输入时为右端矩阵 B，输出时被覆盖为解矩阵 X。各 `Barray[i]` 之间不应重叠 |
| `ldb` | Host | 输入 | `int` | 每个矩阵 `Barray[i]` 的 leading dimension，ldb >= max(1, n) |
| `info` | Host | 输出 | `int *` | 仅反映参数级错误：`*info == 0` 表示参数校验通过；`*info = -j` 表示第 j 个参数非法（j 从 1 开始计数）。**不提供 per-batch 奇异信息**，奇异矩阵检测应在 `aclblasSgetrfBatched` 阶段完成 |
| `batchCount` | Host | 输入 | `int` | 指针数组中包含的矩阵数量，batchCount >= 0 |

**注意**：
- `Aarray` 和 `Barray` 均为 **Device 侧指针数组**（非 Host 侧），调用者需先在 Device 内存中分配指针数组并填充各矩阵地址
- 矩阵以 **列主序**（Column-major）存储，与 LAPACK 标准一致
- 本函数为异步执行（Kernel 通过 handle 的 stream 提交），用户需自行通过 `aclrtSynchronizeStream` 同步
- 调用本函数前，必须先使用 `aclblasSgetrfBatched` 对 `Aarray[i]` 完成 LU 分解
- 典型调用模式：`aclblasSgetrfBatched` 的输出 `PivotArray` 应作为 `aclblasSgetrsBatched` 的输入 `devIpiv` 传入，两者共享相同的 n、Aarray、lda、batchCount 参数

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` 不是 `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` 之一 | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `nrhs < 0` 或 `nrhs > 256` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` 或 `nrhs == 0` 或 `batchCount == 0` | `ACLBLAS_STATUS_SUCCESS`（直接返回，不启动 Kernel） |
| `Aarray == nullptr`（当 batchCount > 0 且 n > 0 且 nrhs > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `Barray == nullptr`（当 batchCount > 0 且 n > 0 且 nrhs > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `devIpiv == nullptr` | 合法（表示 LU 分解未使用主元选取，P = I） |

### 返回值

| 错误码 | 说明 |
|--------|------|
| `ACLBLAS_STATUS_SUCCESS` | 操作成功完成（含 n=0 / nrhs=0 / batchCount=0 的空操作） |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACLBLAS_STATUS_INVALID_ENUM` | trans 枚举值非法 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数非法（见参数约束表） |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 内部错误（如获取核数失败） |

### 实现策略

| 维度 | 策略 | 说明 |
|------|------|------|
| 编程模型 | SIMT | 不规则内存访问（指针数组间接寻址、行置换、三角求解的列间顺序依赖）适合 SIMT 的 GM 直接访问模式 |
| 多核并行 | Batch 维度均匀分配 | batch 均匀分配到 AI Core，每个 Core 处理 batchPerCore 个 batch |
| 线程并行 | 256 SIMT 线程 | 每个线程处理一列右端项（threadIdx.x < nrhs），nrhs 较小时部分线程空闲 |
| 数据存储 | 全程 GM 直接读写 | 矩阵数据在 GM 上操作，通过 asc_syncthreads 做线程间同步，DCache 自动缓存热数据 |
| 小矩阵优化 | 寄存器缓存（n ≤ 8） | 小矩阵场景将整列数据缓存到寄存器，消除 GM 读写冒险 |
| 转置模式 | OP_N / OP_T / OP_C | trans=N 走正向求解路径；trans=T/C 走转置求解路径（实数域等价） |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) + INT32（devIpiv / info） |
| 目标芯片 | Ascend950PR / Ascend950DT（编译时 `--soc=ascend950` 覆盖两者） |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMT |

## 目录结构

```
blas/getrs_batched/
├── README.md                                         // 本文档
└── arch35/
    ├── sgetrs_batched_host.cpp                       // Host 侧：参数校验、TilingData 计算、Kernel 启动
    ├── sgetrs_batched_kernel.cpp                     // Kernel 侧：SIMT 批量线性方程组求解（主元/非主元、转置/非转置四路径）
    ├── sgetrs_batched_kernel.h                       // Kernel 启动函数声明（算子本地头文件）
    └── sgetrs_batched_tiling_data.h                  // TilingData 结构体（Host 和 Kernel 共用）
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子
bash build.sh --ops=getrs_batched --soc=ascend950
```

## 测试

```bash
# 编译并运行精度测试
bash build.sh --ops=getrs_batched --soc=ascend950 --run
```

## 调用示例

以下示例演示完整的批量线性方程组求解流程：先调用 `aclblasSgetrfBatched` 进行 LU 分解，再调用 `aclblasSgetrsBatched` 求解方程组。

对 2 个 3×3 矩阵执行批量求解（带主元选取，trans=N，nrhs=1）：

- **矩阵 A[0]**：非奇异矩阵，求解成功
- **矩阵 A[1]**：对角矩阵，求解成功

> 奇异矩阵行为：若输入矩阵奇异（U 的对角元为 0），求解结果未定义（可能包含 inf/nan），`info` 不上报奇异信息。应在 `aclblasSgetrfBatched` 阶段通过 `infoArray` 检测奇异。

```cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// 示例: 批量线性方程组求解 (batchCount=2, n=3, nrhs=1, trans=N)
//
//   A[0] = [ 2  1  1 ]    A[1] = [ 1  0  0 ]
//          [ 4  3  3 ]           [ 0  2  0 ]
//          [ 6  5  4 ]           [ 0  0  4 ]
//
//   B[0] = [ 4  ]    B[1] = [ 1  ]
//          [ 10 ]           [ 4  ]
//          [ 15 ]           [ 8  ]
//
//   列主序存储:
//     A[0] = {2, 4, 6,  1, 3, 5,  1, 3, 4}
//     A[1] = {1, 0, 0,  0, 2, 0,  0, 0, 4}
//     B[0] = {4, 10, 15}
//     B[1] = {1, 4, 8}
//
//   期望结果:
//     X[0] = A[0]^{-1} * B[0]（求解成功）
//     X[1] = A[1]^{-1} * B[1] = {1, 2, 2}（对角矩阵）
// ==========================================================================

int main()
{
    constexpr int n = 3;
    constexpr int nrhs = 1;
    constexpr int lda = 3;
    constexpr int ldb = 3;
    constexpr int batchCount = 2;
    constexpr size_t matSize = n * lda * sizeof(float);
    constexpr size_t rhsSize = n * nrhs * sizeof(float);

    // Host 侧矩阵数据 (列主序)
    float hA0[n * lda] = {2.0f, 4.0f, 6.0f,  1.0f, 3.0f, 5.0f,  1.0f, 3.0f, 4.0f};
    float hA1[n * lda] = {1.0f, 0.0f, 0.0f,  0.0f, 2.0f, 0.0f,  0.0f, 0.0f, 4.0f};

    // Host 侧右端矩阵 (列主序, n×nrhs)
    float hB0[n * nrhs] = {4.0f, 10.0f, 15.0f};
    float hB1[n * nrhs] = {1.0f, 4.0f, 8.0f};

    int hInfo = 0;
    int ret = 0;

    // Device 资源（统一在 cleanup 释放）
    float *dA0 = nullptr, *dA1 = nullptr;
    const float **dAPtrArray = nullptr;
    int *dPivot = nullptr, *dInfoArray = nullptr;
    float *dB0 = nullptr, *dB1 = nullptr;
    float **dBPtrArray = nullptr;
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;

    // 1. 初始化 ACL 运行环境
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 2. 创建 handle 并绑定 stream
    if (aclblasCreate(&handle) != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasCreate failed" << std::endl;
        ret = -1; goto cleanup;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed" << std::endl;
        ret = -1; goto cleanup;
    }
    if (aclblasSetStream(handle, stream) != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasSetStream failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 3. 在 Device 侧分配矩阵 A 内存并拷贝数据
    if (aclrtMalloc(reinterpret_cast<void **>(&dA0), matSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&dA1), matSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc for A matrices failed" << std::endl;
        ret = -1; goto cleanup;
    }
    if (aclrtMemcpy(dA0, matSize, hA0, matSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(dA1, matSize, hA1, matSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy for A matrices failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 4. 在 Device 侧构建 A 的输入指针数组 (Aarray 本身在 Device 内存中)
    {
        const float* hAPtrArray[batchCount] = {dA0, dA1};
        if (aclrtMalloc(reinterpret_cast<void **>(&dAPtrArray),
                    batchCount * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc for A pointer array failed" << std::endl;
            ret = -1; goto cleanup;
        }
        if (aclrtMemcpy(dAPtrArray, batchCount * sizeof(float*),
                    hAPtrArray, batchCount * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy for A pointer array failed" << std::endl;
            ret = -1; goto cleanup;
        }
    }

    // 5. 分配 PivotArray 和 infoArray (用于 LU 分解)
    if (aclrtMalloc(reinterpret_cast<void **>(&dPivot),
                batchCount * n * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&dInfoArray),
                batchCount * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc for pivot/info arrays failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 6. 调用 aclblasSgetrfBatched 进行 LU 分解
    {
        float** dMutableAPtrArray = const_cast<float**>(dAPtrArray);
        aclblasStatus_t status = aclblasSgetrfBatched(
            handle,
            n,                       // n          — 方阵边长
            dMutableAPtrArray,       // Aarray     — Device 侧指针数组 (LU 分解原地修改)
            lda,                     // lda        — leading dimension
            dPivot,                  // PivotArray — 主元序列输出
            dInfoArray,              // infoArray  — 分解结果信息
            batchCount);             // batchCount — 批量大小

        if (status != ACLBLAS_STATUS_SUCCESS) {
            std::cerr << "aclblasSgetrfBatched failed, status = " << status << std::endl;
            ret = -1; goto cleanup;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "aclrtSynchronizeStream failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 检查 LU 分解结果
    {
        int hInfoCheck[batchCount] = {};
        if (aclrtMemcpy(hInfoCheck, batchCount * sizeof(int),
                    dInfoArray, batchCount * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy for info check failed" << std::endl;
            ret = -1; goto cleanup;
        }
        for (int b = 0; b < batchCount; b++) {
            if (hInfoCheck[b] != 0) {
                std::cerr << "LU decomposition failed for batch " << b
                          << ": U(" << hInfoCheck[b] << "," << hInfoCheck[b] << ") == 0" << std::endl;
                ret = -1; goto cleanup;
            }
        }
    }

    // 7. 在 Device 侧分配 B 矩阵内存并拷贝右端项数据
    if (aclrtMalloc(reinterpret_cast<void **>(&dB0), rhsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&dB1), rhsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc for B matrices failed" << std::endl;
        ret = -1; goto cleanup;
    }
    if (aclrtMemcpy(dB0, rhsSize, hB0, rhsSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(dB1, rhsSize, hB1, rhsSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy for B matrices failed" << std::endl;
        ret = -1; goto cleanup;
    }

    // 8. 在 Device 侧构建 B 的输入指针数组
    {
        float* hBPtrArray[batchCount] = {dB0, dB1};
        if (aclrtMalloc(reinterpret_cast<void **>(&dBPtrArray),
                    batchCount * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc for B pointer array failed" << std::endl;
            ret = -1; goto cleanup;
        }
        if (aclrtMemcpy(dBPtrArray, batchCount * sizeof(float*),
                    hBPtrArray, batchCount * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy for B pointer array failed" << std::endl;
            ret = -1; goto cleanup;
        }
    }

    // 9. 调用 aclblasSgetrsBatched 求解线性方程组
    {
        aclblasStatus_t status = aclblasSgetrsBatched(
            handle,
            ACLBLAS_OP_N,            // trans      — 不转置
            n,                       // n          — 方阵边长
            nrhs,                    // nrhs       — 右端项列数
            dAPtrArray,              // Aarray     — LU 分解结果 (Device 侧指针数组)
            lda,                     // lda        — A 的 leading dimension
            dPivot,                  // devIpiv    — 主元序列 (NULL 则为无主元模式)
            dBPtrArray,              // Barray     — 右端项输入/解输出 (Device 侧指针数组)
            ldb,                     // ldb        — B 的 leading dimension
            &hInfo,                  // info       — 求解结果信息 (Host 侧输出)
            batchCount);             // batchCount — 批量大小

        if (status != ACLBLAS_STATUS_SUCCESS) {
            std::cerr << "aclblasSgetrsBatched failed, status = " << status << std::endl;
            ret = -1; goto cleanup;
        }
    }

    // 10. 同步并读取结果
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "aclrtSynchronizeStream failed" << std::endl;
        ret = -1; goto cleanup;
    }

    std::cout << "info = " << hInfo << std::endl;

    {
        float hX0[n * nrhs], hX1[n * nrhs];
        if (aclrtMemcpy(hX0, rhsSize, dB0, rhsSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            aclrtMemcpy(hX1, rhsSize, dB1, rhsSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy for results failed" << std::endl;
            ret = -1; goto cleanup;
        }

        // 11. 打印结果
        std::cout << "\nSolution X[0] (column-major):" << std::endl;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < nrhs; col++) {
                std::cout << hX0[row + col * ldb] << "\t";
            }
            std::cout << std::endl;
        }

        std::cout << "\nSolution X[1] (column-major):" << std::endl;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < nrhs; col++) {
                std::cout << hX1[row + col * ldb] << "\t";
            }
            std::cout << std::endl;
        }
    }

cleanup:
    // 12. 资源释放（aclrtFree 对 nullptr 安全，无需判空）
    aclrtFree(dA0);
    aclrtFree(dA1);
    aclrtFree(dB0);
    aclrtFree(dB1);
    aclrtFree(dAPtrArray);
    aclrtFree(dBPtrArray);
    aclrtFree(dPivot);
    aclrtFree(dInfoArray);
    if (stream) aclrtDestroyStream(stream);
    if (handle) aclblasDestroy(handle);
    aclFinalize();

    return 0;
}

// 期望输出:
//   info = 0
//
//   Solution X[0] (column-major):
//   1       (近似值)
//   1       (近似值)
//   1       (近似值)
//
//   Solution X[1] (column-major):
//   1
//   2
//   2
```

### 转置模式示例（trans=T）

将 `trans` 设为 `ACLBLAS_OP_T` 即可求解 A^T * X = B：

```cpp
aclblasStatus_t status = aclblasSgetrsBatched(
    handle,
    ACLBLAS_OP_T,            // trans = T → 求解 A^T * X = B
    n,
    nrhs,
    dAPtrArray,
    lda,
    dPivot,                  // devIpiv (NULL 则为无主元模式)
    dBPtrArray,
    ldb,
    &hInfo,
    batchCount);
```

### 非主元模式示例

将 `devIpiv` 设为 `NULL` 即可使用无主元模式（前提是 LU 分解时也未使用主元选取）：

```cpp
aclblasStatus_t status = aclblasSgetrsBatched(
    handle,
    ACLBLAS_OP_N,
    n,
    nrhs,
    dAPtrArray,
    lda,
    nullptr,                 // devIpiv = NULL → 无主元模式 (P = I)
    dBPtrArray,
    ldb,
    &hInfo,
    batchCount);
```

非主元模式跳过行置换步骤，适用于已知矩阵不需要主元选取的场景（如对角占优矩阵）。
