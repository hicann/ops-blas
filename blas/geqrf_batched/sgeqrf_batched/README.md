# aclblasSgeqrfBatched

批量 QR 分解（Batched QR Factorization via Householder Reflections）。对每个批次 `j`（`j = 0, 1, ..., batchSize-1`），使用 Householder 反射将 `m×n` 实矩阵 `A[j]` 分解为 `A[j] = Q[j] * R[j]`。

输出格式：
- `A[j]` 的上三角部分存储 `R[j]`
- `A[j]` 的下三角部分（不含对角线）存储 Householder 向量 `v` 的非平凡部分
- `TauArray[j]` 存储各反射的标量因子 `tau`

## 接口

```c
aclblasStatus_t aclblasSgeqrfBatched(
    aclblasHandle_t handle,
    int m,
    int n,
    float *const Aarray[],
    int lda,
    float *const TauArray[],
    int *info,
    int batchSize);
```

### 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| handle | Host | 输入 | aclblasHandle_t | ops-blas 库上下文句柄 |
| m | Host | 输入 | int | 每个矩阵 Aarray[i] 的行数，要求 m ≥ 0 |
| n | Host | 输入 | int | 每个矩阵 Aarray[i] 的列数，要求 n ≥ 0 |
| Aarray | Device | 输入/输出 | float *const [] | 设备端指针数组，每个元素指向一个 m×n 列主序矩阵。输入时包含原始矩阵，输出时下三角存储 Householder 向量 v，上三角存储 R |
| lda | Host | 输入 | int | 矩阵 Aarray[i] 的前导维度（leading dimension），要求 lda ≥ max(1, m) |
| TauArray | Device | 输出 | float *const [] | 设备端指针数组，每个元素指向维度 min(m, n) 的向量，存储 Householder 标量因子 tau |
| info | Host | 输出 | int* | Host 端指针，指向单个 int。0 = 成功 |
| batchSize | Host | 输入 | int | Aarray 中包含的指针数量（批次数），要求 batchSize ≥ 0 |

### 返回值

| 返回值 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 执行成功 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空指针 |
| ACLBLAS_STATUS_INVALID_VALUE | 参数不合法（m < 0、n < 0、lda < max(1,m)、batchSize < 0、指针为空等） |
| ACLBLAS_STATUS_ALLOC_FAILED | 内存分配失败 |
| ACLBLAS_STATUS_EXECUTION_FAILED | Kernel 执行或流同步失败 |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) |
| 目标芯片 | Ascend950 |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMT |
| CANN 版本 | 9.0.0-beta.2 |

## 编译

```bash
bash build.sh --ops=geqrf_batched
```

## 测试

```bash
bash build.sh --ops=geqrf_batched --run
```

## 调用示例

```cpp
#include <cstdio>
#include <cstdlib>
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    // 1. 初始化 ACL 和 ops-blas 句柄
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    // 2. 设置矩阵参数
    int m = 4;
    int n = 3;
    int lda = m;
    int batchSize = 2;
    int info = 0;

    // 3. 在 Host 端准备矩阵数据
    size_t matrixBytes = static_cast<size_t>(lda) * n * sizeof(float);
    size_t tauBytes = static_cast<size_t>((m < n ? m : n)) * sizeof(float);

    // Host 端矩阵数据（列主序）
    float h_A0[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                       5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f};
    float h_A1[12] = {12.0f, 11.0f, 10.0f, 9.0f,
                       8.0f, 7.0f, 6.0f, 5.0f,
                       4.0f, 3.0f, 2.0f, 1.0f};

    // 4. 在 Device 端分配矩阵和 Tau 内存
    float *d_A0 = nullptr, *d_A1 = nullptr;
    float *d_Tau0 = nullptr, *d_Tau1 = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&d_A0), matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_A1), matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_Tau0), tauBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_Tau1), tauBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    // 拷贝矩阵数据到 Device
    aclrtMemcpy(d_A0, matrixBytes, h_A0, matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_A1, matrixBytes, h_A1, matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 在 Device 端构建指针数组
    float* h_Aarray[2] = {d_A0, d_A1};
    float* h_TauArray[2] = {d_Tau0, d_Tau1};

    float** d_Aarray = nullptr;
    float** d_TauArray = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&d_Aarray), batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_TauArray), batchSize * sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_Aarray, batchSize * sizeof(float*), h_Aarray,
                batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_TauArray, batchSize * sizeof(float*), h_TauArray,
                batchSize * sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);

    // 6. 调用 aclblasSgeqrfBatched
    aclblasStatus_t status = aclblasSgeqrfBatched(
        handle, m, n,
        const_cast<float* const*>(d_Aarray), lda,
        const_cast<float* const*>(d_TauArray),
        &info, batchSize);

    if (status != ACLBLAS_STATUS_SUCCESS) {
        printf("aclblasSgeqrfBatched failed, status=%d, info=%d\n", status, info);
    } else {
        printf("aclblasSgeqrfBatched succeeded, info=%d\n", info);
    }

    // 7. 拷贝结果回 Host 并打印
    float h_R0[12] = {};
    float h_Tau0_out[3] = {};
    aclrtMemcpy(h_R0, matrixBytes, d_A0, matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(h_Tau0_out, tauBytes, d_Tau0, tauBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    printf("Batch 0 - R (upper triangle):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.4f ", (j >= i) ? h_R0[i + j * lda] : 0.0f);
        }
        printf("\n");
    }
    printf("Batch 0 - Tau: ");
    for (int i = 0; i < (m < n ? m : n); i++) {
        printf("%.6f ", h_Tau0_out[i]);
    }
    printf("\n");

    // 8. 释放资源
    aclrtFree(d_A0);
    aclrtFree(d_A1);
    aclrtFree(d_Tau0);
    aclrtFree(d_Tau1);
    aclrtFree(d_Aarray);
    aclrtFree(d_TauArray);

    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

## 文件结构

```
blas/geqrf_batched/sgeqrf_batched/
├── README.md                                    # 本文档
└── arch35/
    ├── sgeqrf_batched_host.cpp                  # Host 侧：参数校验、Tiling 计算、Kernel 启动
    ├── sgeqrf_batched_kernel.cpp                # Device 侧：SIMT Householder QR 分解 Kernel
    └── sgeqrf_batched_tiling_data.h             # TilingData 结构体定义
```

## 设计要点

- **多核策略**：批次维度均匀分配到多个 AI Core，`batchPerCore = ceil(batchSize / coreNum)`，最后一个 Core 处理余数
- **SIMT 线程并行**：每个 Core 内使用 2048 线程并行处理行维度的向量操作（内积、归一化、秩-1 更新）
- **DCache 自动管理**：SIMT 模式下 GM 访问通过 DCache（128 KB）自动缓存
- **列间顺序执行**：Householder 反射具有列间依赖，单矩阵内 k = min(m,n) 步按列顺序串行执行
- **tau=0 快速路径**：当 σ=0 且 x₁≥0 时跳过归一化和秩-1 更新，直接写回结果
