# aclblasSgelsBatched

## 接口

```c
aclblasStatus_t aclblasSgelsBatched(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m,
    int n,
    int nrhs,
    float *const Aarray[],
    int lda,
    float *const Carray[],
    int ldc,
    int *devInfo,
    int batchSize);
```

## 功能

批量求解线性最小二乘/最小范数问题。对每个批次 i（0 ≤ i < batchSize），独立求解：

```
当 trans == ACLBLAS_OP_N 时：
  超定 (m >= n): min || C[i] - A[i] * X ||_2    → QR 分解: A = Q*R, X = R^{-1} * Q^T * C
  欠定 (m <  n): min || X ||_2, s.t. A[i]*X = C → LQ 分解: A = L*Q, X = Q^T * L^{-1} * C

当 trans == ACLBLAS_OP_T 时：
  将 A[i] 替换为 A[i]^T，即求解 A[i]^T * X = C[i] 的最小二乘/最小范数解。
  Host 侧将 OP_T 转换为 OP_N（转置 A 并交换 m/n），Kernel 统一按 OP_N 处理。
```

算法来源：LAPACK SGELS（sgels.f），采用 Householder 反射实现 QR/LQ 分解。接口签名严格对齐 cuBLAS `cublasSgelsBatched`。

### 参数说明

| 参数 | 方向 | 位置 | 说明 |
|------|------|------|------|
| handle | in | Host | ops-blas 库上下文句柄，内部携带 stream |
| trans | in | Host | 操作类型：`ACLBLAS_OP_N(111)` — 不转置；`ACLBLAS_OP_T(112)` — 转置。实数类型不支持 `ACLBLAS_OP_C` |
| m | in | Host | 矩阵 A[i] 的行数，m ≥ 0 |
| n | in | Host | 矩阵 A[i] 的列数，n ≥ 0 |
| nrhs | in | Host | 右端项个数（C[i] 的列数），nrhs ≥ 0 |
| Aarray | in/out | Device | 设备指针数组，含 batchSize 个指针，每个指向 m×n 的 float 矩阵（列主序）。分解后 A 被覆盖为 QR/LQ 因子 |
| lda | in | Host | A[i] 的 leading dimension，lda ≥ max(1, m) |
| Carray | in/out | Device | 设备指针数组，含 batchSize 个指针，每个指向 max(m,n)×nrhs 的 float 矩阵（列主序）。输入时前 m 行为右端项 b，输出时前 n 行为解 X |
| ldc | in | Host | C[i] 的 leading dimension，ldc ≥ max(1, m, n) |
| devInfo | out | Device | 设备整数数组（长度 batchSize）。`devInfo[i]=0` 表示第 i 批次成功；`devInfo[i]>0` 表示第 i 批次秩亏损（对角元素为零的位置） |
| batchSize | in | Host | 批次数量，batchSize ≥ 0 |

**注意**：Aarray、Carray、devInfo 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T` | `ACLBLAS_STATUS_INVALID_ENUM` |
| `m < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `nrhs < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchSize < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldc < max(1, m, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray == nullptr`（当 batchSize > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `Carray == nullptr`（当 batchSize > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `devInfo == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m==0 \|\| n==0 \|\| nrhs==0 \|\| batchSize==0` | `ACLBLAS_STATUS_SUCCESS`（零维度快速返回） |

### 算法路径

| 场景 | 条件 | 算法 |
|------|------|------|
| 超定 + 不转置 | m ≥ n, trans == OP_N | QR 分解 → 应用 Q^T 到 C → 回代求解 R*X = C' |
| 欠定 + 不转置 | m < n, trans == OP_N | LQ 分解 → 前代求解 L*Y = C → 清零 C[m:n,:] → 应用 Q^T |
| 超定 + 转置 | m ≥ n, trans == OP_T | Host 侧转置 A 并交换 m/n，转为 OP_N 欠定问题 |
| 欠定 + 转置 | m < n, trans == OP_T | Host 侧转置 A 并交换 m/n，转为 OP_N 超定问题 |
| 秩亏损 | R/L 对角元素 = 0 | 设置 `devInfo[i] = k+1`，跳过该批次求解 |

### 实现特征

| 特征 | 说明 |
|------|------|
| 编程模型 | SIMT（扁平化结构，128 线程并行） |
| 多核调度 | 按 batch 维度均匀分配到多个 Vector Core，`batchPerCore = ceil(batchSize / usedCoreNum)` |
| 线程并行 | 128 线程协作完成点积、axpy、范数计算等向量操作，块级二分归约 |
| Kernel 结构 | 单次 Kernel 启动处理所有批次，Kernel 内部循环遍历分配给当前 Core 的批次 |
| 数据访问 | 矩阵 A/C 在 GM 中通过 `__gm__` 指针直接读写，列主序存储 |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 |
| 目标芯片 | Ascend950（Ascend950PR / Ascend950DT） |
| 目标架构 | arch35 (DAV_3510) |

## 目录结构

```
blas/gels_batched/
└── sgels_batched/
    ├── README.md
    └── arch35/
        ├── gels_batched_host.cpp          # Host 侧 API（参数校验、OP_T 转置、指针数组扁平化、Kernel 调用）
        ├── gels_batched_kernel.cpp        # Kernel 入口 + SIMT VF 函数（QR/LQ 分解、应用 Q、三角求解）
        └── gels_batched_tiling_data.h     # TilingData 结构体（多核分配参数）
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=gels_batched --soc=ascend950

# 编译并运行测试
bash build.sh --ops=gels_batched --soc=ascend950 --run
```

## 调用示例

以下示例使用 2 个批次求解超定线性最小二乘问题（m=4, n=2, nrhs=1），即 min ||Ax - b||_2：

```cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// 示例: 批量求解超定最小二乘问题 (m=4, n=2, nrhs=1, batchSize=2)
//
// 批次 0:                          批次 1:
//   A0 = [1 1]                       A1 = [2 0]
//        [1 2]                            [0 2]
//        [1 3]                            [1 1]
//        [1 4]                            [1 2]
//   b0 = [1 2 3 4]^T                b1 = [4 6 5 8]^T
//
// 最小二乘解:                       最小二乘解:
//   x0 ≈ [-0.3, 0.8]^T               x1 ≈ [1.0, 2.0]^T (近似)
// ==========================================================================

int main()
{
    constexpr int m = 4;
    constexpr int n = 2;
    constexpr int nrhs = 1;
    constexpr int lda = m;
    constexpr int ldc = m;  // ldc >= max(m, n) = 4
    constexpr int batchSize = 2;

    // 矩阵 A（列主序）
    // 批次 0: A0 = [[1,1],[1,2],[1,3],[1,4]]
    float hA0[m * n] = {
        1.0f, 1.0f, 1.0f, 1.0f,   // 第1列
        1.0f, 2.0f, 3.0f, 4.0f    // 第2列
    };
    // 批次 1: A1 = [[2,0],[0,2],[1,1],[1,2]]
    float hA1[m * n] = {
        2.0f, 0.0f, 1.0f, 1.0f,   // 第1列
        0.0f, 2.0f, 1.0f, 2.0f    // 第2列
    };

    // 右端项 C（列主序，大小为 ldc × nrhs，前 m 行有效）
    float hC0[ldc * nrhs] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hC1[ldc * nrhs] = {4.0f, 6.0f, 5.0f, 8.0f};

    // 1. 初始化 ASCEND 运行环境
    aclInit(nullptr);

    // 2. 创建 handle 并绑定 stream
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    // 3. 分配 device 内存并拷贝数据
    float *dA0 = nullptr, *dA1 = nullptr;
    float *dC0 = nullptr, *dC1 = nullptr;
    int *dDevInfo = nullptr;

    aclrtMalloc(reinterpret_cast<void **>(&dA0), m * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dA1), m * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dC0), ldc * nrhs * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dC1), ldc * nrhs * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dDevInfo), batchSize * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA0, m * n * sizeof(float), hA0, m * n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dA1, m * n * sizeof(float), hA1, m * n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dC0, ldc * nrhs * sizeof(float), hC0, ldc * nrhs * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dC1, ldc * nrhs * sizeof(float), hC1, ldc * nrhs * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 构建设备指针数组（存储在 device 内存中）
    float* hPtrsA[batchSize] = {dA0, dA1};
    float* hPtrsC[batchSize] = {dC0, dC1};

    float** dAarray = nullptr;
    float** dCarray = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dAarray), batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dCarray), batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dAarray, batchSize * sizeof(float*), hPtrsA, batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dCarray, batchSize * sizeof(float*), hPtrsC, batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 调用 aclblasSgelsBatched
    aclblasStatus_t status = aclblasSgelsBatched(
        handle,
        ACLBLAS_OP_N,         // trans     — 不转置
        m,                    // m         — A 的行数
        n,                    // n         — A 的列数
        nrhs,                 // nrhs      — 右端项个数
        dAarray,              // Aarray    — 设备指针数组 (A)
        lda,                  // lda       — A 的 leading dimension
        dCarray,              // Carray    — 设备指针数组 (C)
        ldc,                  // ldc       — C 的 leading dimension
        dDevInfo,             // devInfo   — 每批次求解状态
        batchSize);           // batchSize — 批次数量

    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasSgelsBatched failed, status = " << status << std::endl;
        return -1;
    }

    // 6. 同步并拷贝结果回 host
    aclrtSynchronizeStream(stream);

    int hDevInfo[batchSize] = {};
    aclrtMemcpy(hDevInfo, batchSize * sizeof(int), dDevInfo, batchSize * sizeof(int),
                ACL_MEMCPY_DEVICE_TO_HOST);

    float hC0Out[ldc * nrhs] = {};
    float hC1Out[ldc * nrhs] = {};
    aclrtMemcpy(hC0Out, ldc * nrhs * sizeof(float), dC0, ldc * nrhs * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hC1Out, ldc * nrhs * sizeof(float), dC1, ldc * nrhs * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    // 7. 打印结果
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << ": devInfo = " << hDevInfo[b] << std::endl;
    }

    std::cout << "Batch 0 solution (x):" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "  x[" << i << "] = " << hC0Out[i] << std::endl;
    }

    std::cout << "Batch 1 solution (x):" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "  x[" << i << "] = " << hC1Out[i] << std::endl;
    }

    // 8. 资源释放
    aclrtFree(dAarray);
    aclrtFree(dCarray);
    aclrtFree(dA0);
    aclrtFree(dA1);
    aclrtFree(dC0);
    aclrtFree(dC1);
    aclrtFree(dDevInfo);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclFinalize();

    return 0;
}

// 期望输出:
//   Batch 0: devInfo = 0
//   Batch 1: devInfo = 0
//   Batch 0 solution (x):
//     x[0] = -0.3
//     x[1] = 0.8
//   Batch 1 solution (x):
//     x[0] = 1.0
//     x[1] = 2.0
```
