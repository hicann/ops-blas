# aclblasSgetrfBatched

## 接口

```c
aclblasStatus_t aclblasSgetrfBatched(
    aclblasHandle_t handle,
    int n,
    float *const Aarray[],
    int lda,
    int *PivotArray,
    int *infoArray,
    int batchSize);
```

## 功能

对一批 n×n 方阵分别执行 **带部分主元选取的 LU 分解**（LU factorization with partial pivoting）：

```
P * A[i] = L * U,   i = 0, 1, ..., batchSize - 1
```

| 符号 | 含义 |
|------|------|
| `A[i]` | 第 i 个 n×n 方阵（列主序存储，leading dimension = `lda`） |
| `P` | 置换矩阵，由 `PivotArray[i]` 中的主元序列构造 |
| `L` | 单位下三角矩阵（对角线为 1，不存储） |
| `U` | 上三角矩阵 |

分解完成后，L（不含对角线 1）和 U 被原地写回 `Aarray[i]` 的存储空间。

**L/U 存储约定**：行交换时交换完整行（包括列 0 到 col-1 中已计算的 L 乘子），因此输出的 L 因子与标准 LAPACK sgetrf 的存储布局一致，可直接用于 LAPACK 兼容的 sgetrs/sgetri 求解。

如果 `PivotArray` 为 `NULL`，则禁用主元选取，执行非主元 LU 分解（直接使用对角元素作为主元，不做行交换）。

> 对齐 cuBLAS `cublasSgetrfBatched`（[官方文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrfbatched)），算法参考 NETLIB LAPACK sgetrf。

### 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| `handle` | Host | 输入 | `aclblasHandle_t` | ops-blas 库上下文句柄，内部携带 stream |
| `n` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的行数和列数（方阵边长），n >= 0 |
| `Aarray` | Device | 输入/输出 | `float *const []` | **Device 侧指针数组**：指针数组本身存储在 Device 内存中，每个元素 `Aarray[i]` 是一个指向 Device 内存中 n×n float 矩阵的指针。矩阵以列主序存储，leading dimension = `lda`。各矩阵之间不应重叠 |
| `lda` | Host | 输入 | `int` | 每个矩阵 `Aarray[i]` 的 leading dimension，lda >= max(1, n) |
| `PivotArray` | Device | 输出 | `int *` | 大小为 n × batchSize 的数组，存储每个矩阵的主元序列。`PivotArray[i * n + j]` 为第 i 个 batch 第 j 步的行交换索引（1-indexed，LAPACK 约定）。**可为 NULL**，此时禁用主元选取 |
| `infoArray` | Device | 输出 | `int *` | 大小为 batchSize 的数组。`infoArray[i]` = 0 表示分解成功；= k > 0 表示 U(k,k) == 0（U 精确奇异，分解已完成但不可用于求解）。当 `PivotArray != NULL` 时 `infoArray` 不可为 NULL；非主元模式下 `infoArray` 可为 NULL |
| `batchSize` | Host | 输入 | `int` | 指针数组中包含的矩阵数量，batchSize >= 0 |

**注意**：
- `Aarray` 是 **Device 侧指针数组**（非 Host 侧），调用者需先在 Device 内存中分配指针数组并填充各矩阵地址
- 矩阵以 **列主序**（Column-major）存储，与 LAPACK 标准一致
- 本函数为异步执行（Kernel 通过 handle 的 stream 提交），Host 侧不执行流同步，用户需自行通过 `aclrtSynchronizeStream` 同步
- TilingData 通过值传递（pass-by-value）方式下发至 kernel，无需设备侧内存分配和 H2D 拷贝

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchSize < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` 或 `batchSize == 0` | `ACLBLAS_STATUS_SUCCESS`（直接返回，不启动 Kernel） |
| `Aarray == nullptr`（当 batchSize > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `PivotArray != nullptr` 且 `infoArray == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `PivotArray == nullptr` | 合法（禁用主元选取，执行非主元 LU 分解） |

> 参数约束对齐 cuBLAS `cublasSgetrfBatched` 文档（n < 0、batchSize < 0、lda < max(1,n) 均返回 INVALID_VALUE），`PivotArray == NULL` 时禁用主元选取的行为亦来自 cuBLAS 文档。

### 返回值

| 错误码 | 说明 |
|--------|------|
| `ACLBLAS_STATUS_SUCCESS` | 操作成功完成（含 n=0 / batchSize=0 的空操作） |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数非法（见参数约束表） |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 内部错误（如获取核数失败） |

> 错误码映射自 cuBLAS 文档（`CUBLAS_STATUS_SUCCESS` / `CUBLAS_STATUS_NOT_INITIALIZED` / `CUBLAS_STATUS_INVALID_VALUE` / `CUBLAS_STATUS_EXECUTION_FAILED`）。

### 算法流程

对每个矩阵 A[i]，执行高斯消元法（Gaussian Elimination with Partial Pivoting）：

1. 对每列 k = 0, 1, ..., n-1：
   - **选主元**（仅主元模式）：在 A[k:n-1, k] 中找到绝对值最大的元素所在行 p
   - **行交换**（仅主元模式）：交换第 k 行和第 p 行，记录置换到 PivotArray
   - **奇异检查**：若 A[k,k] == 0，则 infoArray[i] = k+1（1-indexed），停止当前矩阵分解
   - **计算乘子**：A[row, k] /= A[k, k]，row = k+1, ..., n-1
   - **矩阵更新**：A[row, c] -= A[row, k] * A[k, c]（rank-1 更新）

2. 分解完成后，L（不含对角线 1）和 U 被写回原矩阵 A 的存储空间

### 实现策略

| 维度 | 策略 | 说明 |
|------|------|------|
| 编程模型 | SIMT | 不规则内存访问（指针数组间接寻址、选主元行交换）适合 SIMT 的 GM 直接访问模式 |
| 多核并行 | Batch 维度均匀分配 | batch 均匀分配到 AI Core，每个 Core 处理 batchPerCore 个 batch |
| 大矩阵支持 | 逐列分解 | 每列依次处理选主元、行交换、乘子计算、rank-1 更新，支持任意大小的 n（不受 UB 容量限制） |
| 数据存储 | 全程 GM 直接读写 | 矩阵数据在 GM 上操作，UB 仅用于线程协作（~2 KB），DCache 自动缓存热数据 |

> 来源：开发方案设计文档 §2（Tiling 策略）+ §3（Kernel 设计）。

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) + int32（PivotArray / infoArray） |
| 目标芯片 | Ascend950PR / Ascend950DT |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMT |

> 来源：需求分析文档 §2.5（Ascend950 → DAV_3510 → arch35，npu-arch skill 映射）。

## 目录结构

```
blas/getrf_batched/
├── README.md                                      // 本文档
└── arch35/
    ├── sgetrf_batched_host.cpp                    // Host 侧：参数校验、TilingData 计算、Kernel 启动
    ├── sgetrf_batched_kernel.cpp                  // Kernel 侧：SIMT LU 分解（主元/非主元双路径）
    └── sgetrf_batched_tiling_data.h               // TilingData 结构体（Host 和 Kernel 共用）
```

测试代码位于 `test/getrf_batched/sgetrf_batched/`：

```
test/getrf_batched/sgetrf_batched/
├── CMakeLists.txt
├── sgetrf_batched_param.h
├── sgetrf_batched_golden.h              // CPU golden（主元模式基于 LAPACK sgetrf_，非主元模式手写实现）
└── arch35/
    ├── sgetrf_batched_test.cpp
    ├── sgetrf_batched_test.csv
    └── sgetrf_batched_npu_wrapper.h
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子
bash build.sh --ops=getrf_batched --soc=ascend950
```

## 测试

```bash
# 编译并运行精度测试
bash build.sh --ops=getrf_batched --soc=ascend950 --run
```

## 调用示例

以下示例对 2 个 3×3 矩阵执行批量 LU 分解（带主元选取）：

- **矩阵 A[0]**：非奇异矩阵，分解成功（infoArray[0] = 0）
- **矩阵 A[1]**：奇异矩阵（第 3 行 = 第 1 行 + 第 2 行），U(2,2) = 0（infoArray[1] = 2）

```cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// 示例: 批量 LU 分解 (batchSize=2, n=3)
//
//   A[0] = [ 2  1  1 ]    A[1] = [ 1  2  3 ]   (奇异: 第3行 = 第1行+第2行)
//          [ 4  3  3 ]           [ 2  4  6 ]
//          [ 6  5  4 ]           [ 3  6  9 ]
//
//   列主序存储:
//     A[0] = {2, 4, 6,  1, 3, 5,  1, 3, 4}
//     A[1] = {1, 2, 3,  2, 4, 6,  3, 6, 9}
//
//   期望结果:
//     A[0]: infoArray[0] = 0 (成功)
//     A[1]: infoArray[1] = 2 (U(2,2) == 0, 1-indexed)
// ==========================================================================

int main()
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int batchSize = 2;
    constexpr size_t matSize = n * lda * sizeof(float);

    // Host 侧矩阵数据 (列主序)
    float hA0[n * lda] = {2.0f, 4.0f, 6.0f,  1.0f, 3.0f, 5.0f,  1.0f, 3.0f, 4.0f};
    float hA1[n * lda] = {1.0f, 2.0f, 3.0f,  2.0f, 4.0f, 6.0f,  3.0f, 6.0f, 9.0f};

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

    // 4. 在 Device 侧构建指针数组 (关键: Aarray 本身也在 Device 内存中)
    float* hPtrArray[batchSize] = {dA0, dA1};
    float **dPtrArray = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dPtrArray),
                batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dPtrArray, batchSize * sizeof(float*),
                hPtrArray, batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 分配 PivotArray 和 infoArray
    int *dPivot = nullptr, *dInfo = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dPivot),
                batchSize * n * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dInfo),
                batchSize * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 调用 aclblasSgetrfBatched
    aclblasStatus_t status = aclblasSgetrfBatched(
        handle,
        n,                    // n        — 方阵边长
        dPtrArray,            // Aarray   — Device 侧指针数组
        lda,                  // lda      — leading dimension
        dPivot,               // PivotArray — 主元序列输出 (NULL 则禁用主元选取)
        dInfo,                // infoArray  — 分解结果信息
        batchSize);           // batchSize  — 批量大小

    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasSgetrfBatched failed, status = " << status << std::endl;
        return -1;
    }

    // 7. 同步并读取结果
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hPivot, batchSize * n * sizeof(int),
                dPivot, batchSize * n * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hInfo, batchSize * sizeof(int),
                dInfo, batchSize * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);

    // 读取分解后的矩阵 (L 和 U 原地存储在 A 中)
    float hA0_out[n * lda], hA1_out[n * lda];
    aclrtMemcpy(hA0_out, matSize, dA0, matSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hA1_out, matSize, dA1, matSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 8. 打印结果
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << ": infoArray[" << b << "] = " << hInfo[b] << std::endl;
        std::cout << "  Pivots: ";
        for (int j = 0; j < n; j++) {
            std::cout << hPivot[b * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // 9. 资源释放
    aclrtFree(dA0);
    aclrtFree(dA1);
    aclrtFree(dPtrArray);
    aclrtFree(dPivot);
    aclrtFree(dInfo);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclFinalize();

    return 0;
}

// 期望输出:
//   Batch 0: infoArray[0] = 0
//     Pivots: 3 3 3
//   Batch 1: infoArray[1] = 2
//     Pivots: 3 2 3
```

### 非主元模式示例

将 `PivotArray` 设为 `NULL` 即可禁用主元选取：

```cpp
aclblasStatus_t status = aclblasSgetrfBatched(
    handle,
    n,
    dPtrArray,
    lda,
    nullptr,      // PivotArray = NULL → 禁用主元选取
    dInfo,        // infoArray 仍可用于检测奇异性（也可为 NULL）
    batchSize);
```

非主元模式下直接使用对角元素作为主元，不做行交换。适用于已知矩阵不需要主元选取的场景（如对角占优矩阵）。
