# aclblasSgetriBatched

## 接口

```c
aclblasStatus_t aclblasSgetriBatched(
    aclblasHandle_t handle,
    int n,
    const float *const Aarray[],
    int lda,
    const int *PivotArray,
    float *const Carray[],
    int ldc,
    int *infoArray,
    int batchSize);
```

## 功能

对一批已经由 `aclblasSgetrfBatched` 完成 LU 分解的 n×n 方阵，批量计算**逆矩阵**：

```
inv(A[i]) = inv(U[i]) * inv(L[i]) * P[i],   i = 0, 1, ..., batchSize - 1
```

| 符号 | 含义 |
|------|------|
| `A[i]` | 第 i 个 n×n 方阵（列主序存储），已由 `aclblasSgetrfBatched` 完成 LU 分解 |
| `P[i]` | 置换矩阵，由 `PivotArray` 中的主元序列构造（`PivotArray` 为 `NULL` 时 P = I） |
| `L[i]` | 单位下三角矩阵（对角线为 1，不显式存储），来自 LU 分解结果 |
| `U[i]` | 上三角矩阵，来自 LU 分解结果 |
| `inv(A[i])` | 第 i 个矩阵的逆矩阵，输出到 `Carray[i]` |

**算法流程**（基于 LAPACK sgetri 标准，使用前后向三角求解器）：

1. 初始化输出矩阵 `C[i] = I`（单位矩阵）
2. 对 `C[i]` 应用置换 `P[i]`（行交换序列）
3. 前向求解：`L[i] * X[i] = P[i] * I`（利用 L 的单位下三角结构逐列前向代入）
4. 后向求解：`U[i] * C[i] = X[i]`（利用 U 的上三角结构逐列后向代入）
5. 最终 `C[i] = inv(A[i])`

若 `U[i](k,k) == 0`（U 精确奇异），则求逆失败，`infoArray[i] = k`（1-indexed）。

> 接口对齐 LAPACK sgetri 标准（[NETLIB 文档](https://www.netlib.org/lapack/explore-html/d8/ddc/sgetri_8f.html)）。

### 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| `handle` | Host | 输入 | `aclblasHandle_t` | ops-blas 库上下文句柄，内部携带 stream |
| `n` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的行数和列数（方阵边长），n >= 0 |
| `Aarray` | Device | 输入 | `const float *const []` | **Device 侧指针数组**：每个元素 `Aarray[i]` 指向 Device 内存中已经 LU 分解的 n×n float 矩阵（列主序，leading dimension = `lda`）。各矩阵之间不应重叠 |
| `lda` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的 leading dimension，lda >= max(1, n) |
| `PivotArray` | Device | 输入 | `const int *` | 大小为 n × batchSize 的数组，存储每个矩阵的主元序列（来自 `aclblasSgetrfBatched` 输出）。`PivotArray[i * n + j]` 为第 i 个 batch 第 j 步的行交换索引（1-indexed，LAPACK 约定）。**可为 NULL**，此时表示 LU 分解未使用主元选取（P = I） |
| `Carray` | Device | 输出 | `float *const []` | **Device 侧指针数组**：每个元素 `Carray[i]` 指向 Device 内存中 n×n float 矩阵（列主序，leading dimension = `ldc`），用于存储逆矩阵。`Carray[i]` 的内存空间不可与 `Aarray[i]` 重叠 |
| `ldc` | Host | 输入 | `int` | 每个矩阵 `Carray[i]` 的 leading dimension，ldc >= max(1, n) |
| `infoArray` | Device | 输出 | `int *` | 大小为 batchSize 的数组。`infoArray[i]` = 0 表示求逆成功；= k > 0 表示 U(k,k) == 0（U 精确奇异，求逆失败） |
| `batchSize` | Host | 输入 | `int` | 指针数组中包含的矩阵数量，batchSize >= 0 |

**注意**：
- `Aarray` 和 `Carray` 均为 **Device 侧指针数组**（非 Host 侧），调用者需先在 Device 内存中分配指针数组并填充各矩阵地址
- 矩阵以 **列主序**（Column-major）存储，与 LAPACK 标准一致
- 本函数为异步执行（Kernel 通过 handle 的 stream 提交），用户需自行通过 `aclrtSynchronizeStream` 同步
- 调用本函数前，必须先使用 `aclblasSgetrfBatched` 对 `Aarray[i]` 完成 LU 分解
- 典型调用模式：`aclblasSgetrfBatched` 和 `aclblasSgetriBatched` 共享相同的 n、Aarray、lda、PivotArray、batchSize 参数

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchSize < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldc < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` 或 `batchSize == 0` | `ACLBLAS_STATUS_SUCCESS`（直接返回，不启动 Kernel） |
| `Aarray == nullptr`（当 batchSize > 0 且 n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `Carray == nullptr`（当 batchSize > 0 且 n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `infoArray == nullptr`（当 batchSize > 0 且 n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `PivotArray == nullptr` | 合法（表示 LU 分解未使用主元选取，P = I） |

### 返回值

| 错误码 | 说明 |
|--------|------|
| `ACLBLAS_STATUS_SUCCESS` | 操作成功完成（含 n=0 / batchSize=0 的空操作） |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数非法（见参数约束表） |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 内部错误（如获取核数失败） |

### 实现策略

| 维度 | 策略 | 说明 |
|------|------|------|
| 编程模型 | SIMT | 不规则内存访问（指针数组间接寻址、行置换、三角求解的列间顺序依赖）适合 SIMT 的 GM 直接访问模式 |
| 多核并行 | Batch 维度均匀分配 | batch 均匀分配到 AI Core，每个 Core 处理 batchPerCore 个 batch |
| 线程并行 | 256 SIMT 线程 | 每个 Core 内 256 线程并行处理行/列维度的向量操作 |
| 数据存储 | 全程 GM 直接读写 | 矩阵数据在 GM 上操作，UB 仅用于线程协作（4 B），DCache 自动缓存热数据 |
| 小矩阵优化 | 寄存器缓存（n ≤ 8） | 小矩阵场景将整列数据缓存到寄存器，消除 GM 读写冒险 |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) + INT32（PivotArray / infoArray） |
| 目标芯片 | Ascend950PR / Ascend950DT |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMT |

## 目录结构

```
blas/getri_batched/
├── README.md                                          // 本文档
└── arch35/
    ├── getri_batched_host.cpp                  // Host 侧：参数校验、TilingData 计算、Kernel 启动
    ├── getri_batched_kernel.cpp                // Kernel 侧：SIMT 批量矩阵求逆（主元/非主元双路径）
    ├── getri_batched_kernel.h                  // Kernel 启动函数声明（算子本地头文件）
    └── getri_batched_tiling_data.h             // TilingData 结构体（Host 和 Kernel 共用）
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子
bash build.sh --ops=getri_batched --soc=ascend950
```

## 测试

```bash
# 编译并运行精度测试
bash build.sh --ops=getri_batched --soc=ascend950 --run
```

## 调用示例

以下示例演示完整的批量矩阵求逆流程：先调用 `aclblasSgetrfBatched` 进行 LU 分解，再调用 `aclblasSgetriBatched` 计算逆矩阵。

对 2 个 3×3 矩阵执行批量求逆（带主元选取）：

- **矩阵 A[0]**：非奇异矩阵，求逆成功（infoArray[0] = 0）
- **矩阵 A[1]**：非奇异矩阵，求逆成功（infoArray[1] = 0）

```cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// 示例: 批量矩阵求逆 (batchSize=2, n=3)
//
//   A[0] = [ 2  1  1 ]    A[1] = [ 1  0  0 ]
//          [ 4  3  3 ]           [ 0  2  0 ]
//          [ 6  5  4 ]           [ 0  0  4 ]
//
//   列主序存储:
//     A[0] = {2, 4, 6,  1, 3, 5,  1, 3, 4}
//     A[1] = {1, 0, 0,  0, 2, 0,  0, 0, 4}
//
//   期望结果:
//     A[0]: infoArray[0] = 0 (求逆成功)
//     A[1]: infoArray[1] = 0 (对角矩阵，求逆成功)
// ==========================================================================

int main()
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int ldc = 3;
    constexpr int batchSize = 2;
    constexpr size_t matSize = n * lda * sizeof(float);

    // Host 侧矩阵数据 (列主序)
    float hA0[n * lda] = {2.0f, 4.0f, 6.0f,  1.0f, 3.0f, 5.0f,  1.0f, 3.0f, 4.0f};
    float hA1[n * lda] = {1.0f, 0.0f, 0.0f,  0.0f, 2.0f, 0.0f,  0.0f, 0.0f, 4.0f};

    int hPivot[batchSize * n] = {};
    int hInfo[batchSize] = {};

    // 1. 初始化 ACL 运行环境
    aclInit(nullptr);

    // 2. 创建 handle 并绑定 stream
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    // 3. 在 Device 侧分配矩阵内存并拷贝数据
    float *dA0 = nullptr, *dA1 = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dA0), matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dA1), matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dA0, matSize, hA0, matSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dA1, matSize, hA1, matSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 在 Device 侧构建输入指针数组 (Aarray 本身在 Device 内存中)
    const float* hInPtrArray[batchSize] = {dA0, dA1};
    const float **dInPtrArray = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dInPtrArray),
                batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dInPtrArray, batchSize * sizeof(float*),
                hInPtrArray, batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 分配 PivotArray 和 infoArray
    int *dPivot = nullptr, *dInfo = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dPivot),
                batchSize * n * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dInfo),
                batchSize * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 调用 aclblasSgetrfBatched 进行 LU 分解
    //    注意: getrf 需要非 const 的 Aarray，此处复用 dInPtrArray 的地址
    float** dMutablePtrArray = const_cast<float**>(dInPtrArray);
    aclblasStatus_t status = aclblasSgetrfBatched(
        handle,
        n,                    // n        — 方阵边长
        dMutablePtrArray,     // Aarray   — Device 侧指针数组 (LU 分解原地修改)
        lda,                  // lda      — leading dimension
        dPivot,               // PivotArray — 主元序列输出
        dInfo,                // infoArray  — 分解结果信息
        batchSize);           // batchSize  — 批量大小

    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasSgetrfBatched failed, status = " << status << std::endl;
        return -1;
    }
    aclrtSynchronizeStream(stream);

    // 检查 LU 分解结果
    aclrtMemcpy(hInfo, batchSize * sizeof(int),
                dInfo, batchSize * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int b = 0; b < batchSize; b++) {
        if (hInfo[b] != 0) {
            std::cerr << "LU decomposition failed for batch " << b
                      << ": U(" << hInfo[b] << "," << hInfo[b] << ") == 0" << std::endl;
            return -1;
        }
    }

    // 7. 分配输出矩阵和输出指针数组
    float *dC0 = nullptr, *dC1 = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dC0), matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dC1), matSize, ACL_MEM_MALLOC_HUGE_FIRST);

    float* hOutPtrArray[batchSize] = {dC0, dC1};
    float **dOutPtrArray = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dOutPtrArray),
                batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dOutPtrArray, batchSize * sizeof(float*),
                hOutPtrArray, batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    // 8. 调用 aclblasSgetriBatched 计算逆矩阵
    status = aclblasSgetriBatched(
        handle,
        n,                    // n        — 方阵边长
        dInPtrArray,          // Aarray   — LU 分解结果 (Device 侧指针数组)
        lda,                  // lda      — A 的 leading dimension
        dPivot,               // PivotArray — 主元序列 (NULL 则为无主元模式)
        dOutPtrArray,         // Carray   — 逆矩阵输出 (Device 侧指针数组)
        ldc,                  // ldc      — C 的 leading dimension
        dInfo,                // infoArray  — 求逆结果信息
        batchSize);           // batchSize  — 批量大小

    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasSgetriBatched failed, status = " << status << std::endl;
        return -1;
    }

    // 9. 同步并读取结果
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hInfo, batchSize * sizeof(int),
                dInfo, batchSize * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);

    float hC0[n * ldc], hC1[n * ldc];
    aclrtMemcpy(hC0, matSize, dC0, matSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hC1, matSize, dC1, matSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 10. 打印结果
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << ": infoArray[" << b << "] = " << hInfo[b] << std::endl;
    }

    std::cout << "\nInverse of A[0] (column-major):" << std::endl;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            std::cout << hC0[row + col * ldc] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nInverse of A[1] (column-major):" << std::endl;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            std::cout << hC1[row + col * ldc] << "\t";
        }
        std::cout << std::endl;
    }

    // 11. 资源释放
    aclrtFree(dA0);
    aclrtFree(dA1);
    aclrtFree(dC0);
    aclrtFree(dC1);
    aclrtFree(dInPtrArray);
    aclrtFree(dOutPtrArray);
    aclrtFree(dPivot);
    aclrtFree(dInfo);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclFinalize();

    return 0;
}

// 期望输出:
//   Batch 0: infoArray[0] = 0
//   Batch 1: infoArray[1] = 0
//
//   Inverse of A[0] (column-major):
//   1.5     -0.5    0
//   -1      -1      1
//   -1      2       -1
//
//   Inverse of A[1] (column-major):
//   1       0       0
//   0       0.5     0
//   0       0       0.25
```

### 非主元模式示例

将 `PivotArray` 设为 `NULL` 即可使用无主元模式（前提是 LU 分解时也未使用主元选取）：

```cpp
aclblasStatus_t status = aclblasSgetriBatched(
    handle,
    n,
    dInPtrArray,
    lda,
    nullptr,          // PivotArray = NULL → 无主元模式 (P = I)
    dOutPtrArray,
    ldc,
    dInfo,
    batchSize);
```

非主元模式跳过行置换步骤，适用于已知矩阵不需要主元选取的场景（如对角占优矩阵）。
