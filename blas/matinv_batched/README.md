# 批量矩阵求逆算子

## 算子概述

批量矩阵求逆算子针对一组 n×n 方阵，分别计算其逆矩阵。核心运算为：

```
Ainv[i] = A[i]⁻¹,  i = 0, 1, ..., batchSize - 1
```

内部通过 LU 分解（PA = LU）加上三角求逆两步完成。支持 n ≤ 32 的小方阵批量求逆。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSmatinvBatched | 单精度批量矩阵求逆（FP32） |

## 算子执行接口

### aclblasSmatinvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSmatinvBatched(
    aclblasHandle_t handle,
    int n,
    const float* const A[],
    int lda,
    float* const Ainv[],
    int lda_inv,
    int* info,
    int batchSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个方阵的行和列数，0 ≤ n ≤ 32，Host 内存 |
| A | 输入 | const float* const[] | Device 侧指针数组，每个元素 `A[i]` 指向第 i 个输入矩阵（FP32，列主序，n×n），Device 内存 |
| lda | 输入 | int | 输入矩阵 A[i] 的 leading dimension，Host 内存 |
| Ainv | 输出 | float* const[] | Device 侧指针数组，每个元素 `Ainv[i]` 指向第 i 个输出矩阵（FP32，列主序，n×n），Device 内存 |
| lda_inv | 输入 | int | 输出矩阵 Ainv[i] 的 leading dimension，Host 内存 |
| info | 输出 | int* | 长度为 batchSize 的数组，`info[i] = 0` 表示求逆成功；`info[i] = k`（k > 0）表示 U(k,k) == 0（矩阵奇异），Device 内存 |
| batchSize | 输入 | int | 矩阵数量，batchSize ≥ 0，Host 内存 |

#### 约束说明

- n >= 0 且 n <= 32，方阵边长
- batchSize >= 0
- lda >= max(1, n)
- lda_inv >= max(1, n)
- 当 n == 0 或 batchSize == 0 时，函数直接返回成功，不访问 A / Ainv / info 指针（允许传 nullptr）
- 当 n > 0 且 batchSize > 0 时，A、Ainv、info 不得为 nullptr
- A[i] 与 Ainv[i] 的内存空间不可重叠
- 矩阵以列主序（Column-major）存储
- 本函数为异步执行，用户需自行通过 `aclrtSynchronizeStream` 同步获取结果
- Workspace：函数内部使用 handle workspace，需求大小为 `batchSize × 8 + n² × batchSize × 4 + n × batchSize × 4` 字节；workspace 不足时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，用户需通过 `aclblasSetWorkspace()` 扩容后重试

#### 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <cstdio>
#include <vector>

int main()
{
    // 1. 初始化
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    // 2. 设置参数
    const int n = 16;
    const int batchSize = 4;
    const int lda = n;
    const int lda_inv = n;

    // 3. 在 Host 侧准备输入矩阵
    std::vector<std::vector<float>> hostA(batchSize);
    std::vector<std::vector<float>> hostAinv(batchSize);
    std::vector<int> hostInfo(batchSize, 0);
    for (int b = 0; b < batchSize; b++) {
        hostA[b].resize((size_t)lda * n, 0.0f);
        hostAinv[b].resize((size_t)lda_inv * n, 0.0f);
        // 列主序初始化为非奇异方阵
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                hostA[b][j * lda + i] = (i == j) ? 2.0f : 0.1f;
            }
        }
    }

    // 4. Device 侧分配矩阵内存并拷贝输入
    std::vector<void*> dAMatrices(batchSize, nullptr);
    std::vector<void*> dAinvMatrices(batchSize, nullptr);
    const size_t matBytes = (size_t)lda * n * sizeof(float);
    const size_t invBytes = (size_t)lda_inv * n * sizeof(float);

    for (int b = 0; b < batchSize; b++) {
        aclrtMalloc(&dAMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dAMatrices[b], matBytes, hostA[b].data(), matBytes,
                     ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMalloc(&dAinvMatrices[b], invBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 5. 构建 Device 侧指针数组（关键步骤）
    const size_t ptrArrayBytes = (size_t)batchSize * sizeof(float*);
    std::vector<float*> hAPtrs(batchSize), hAinvPtrs(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hAPtrs[b] = (float*)dAMatrices[b];
        hAinvPtrs[b] = (float*)dAinvMatrices[b];
    }

    void *dAPtrArray = nullptr, *dAinvPtrArray = nullptr;
    aclrtMalloc(&dAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dAPtrArray, ptrArrayBytes, hAPtrs.data(), ptrArrayBytes,
                 ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMalloc(&dAinvPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dAinvPtrArray, ptrArrayBytes, hAinvPtrs.data(), ptrArrayBytes,
                 ACL_MEMCPY_HOST_TO_DEVICE);

    // 6. Device 侧分配 info 数组
    const size_t infoBytes = (size_t)batchSize * sizeof(int);
    void* dInfo = nullptr;
    aclrtMalloc(&dInfo, infoBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    // 7. 调用接口
    aclblasStatus_t status = aclblasSmatinvBatched(
        handle, n,
        (const float* const*)dAPtrArray, lda,
        (float* const*)dAinvPtrArray, lda_inv,
        (int*)dInfo, batchSize);

    // 8. 同步并拷回结果
    if (status == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasGetStream(handle, &stream);
        aclrtSynchronizeStream(stream);

        for (int b = 0; b < batchSize; b++) {
            aclrtMemcpy(hostAinv[b].data(), invBytes, dAinvMatrices[b], invBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST);
        }
        aclrtMemcpy(hostInfo.data(), infoBytes, dInfo, infoBytes,
                     ACL_MEMCPY_DEVICE_TO_HOST);

        // 9. 检查结果
        for (int b = 0; b < batchSize; b++) {
            if (hostInfo[b] == 0) {
                printf("Batch %d: inversion succeeded\n", b);
            } else {
                printf("Batch %d: singular (info=%d)\n", b, hostInfo[b]);
            }
        }
    } else {
        printf("aclblasSmatinvBatched failed: %d\n", status);
    }

    // 10. 释放 Device 内存
    for (int b = 0; b < batchSize; b++) {
        aclrtFree(dAMatrices[b]);
        aclrtFree(dAinvMatrices[b]);
    }
    aclrtFree(dAPtrArray);
    aclrtFree(dAinvPtrArray);
    aclrtFree(dInfo);

    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```

## 实现概要

采用五步编排方案，在同一 stream 上异步提交五个 kernel（不显式同步，依赖 stream 串行语义）：

```
Step 0: InitPtrArrayKernel  → 在 workspace 中构建指针数组
Step 1: CopyKernel          → Ainv[i] = A[i]（保护输入矩阵不被修改）
Step 2: getrf（复用）        → Ainv[i] 就地 LU 分解
Step 3: CopyKernel          → LU 因子副本存入 workspace 临时缓冲区
Step 4: getri（复用）        → 从 LU 因子计算逆矩阵写入 Ainv[i]
```

**编程模型**：SIMT（`LAUNCH_BOUND(256)`）

**Workspace 三段式布局**：

```
workspace →
+------------------+-------------------------+------------------+
| Zone 0           | Zone 1                  | Zone 2           |
| matTmpPtrArray   | matTmpBuf               | PivotArray       |
| 指针数组          | LU 因子副本 dense blocks | 主元 int 数组     |
| batchSize×8 B    | n²×batchSize×4 B        | n×batchSize×4 B  |
+------------------+-------------------------+------------------+
```

## 性能概要

| shape | dtype | 耗时 (us) | GFLOPS | 吞吐量 (mat/s) |
|-------|-------|---------|--------|-------------|
| n=32, batchSize=1 | FP32 | 376 | 0.17 | 2,657 |
| n=32, batchSize=64 | FP32 | 752 | 5.58 | 85,106 |
| n=32, batchSize=256 | FP32 | 1,860 | 9.02 | 137,634 |
| n=32, batchSize=1024 | FP32 | 7,032 | 9.54 | 145,628 |

测试环境：Ascend950PR，arch35，CANN 9.1.0，54 AI Cores 满频运行。

典型 VEC Bound 算子，VEC 流水线利用率 > 99%。大 batchSize 场景（≥ 64）时核间负载均衡良好。

## 编译

```bash
bash build.sh --ops=matinv_batched
```

## 测试

```bash
bash build.sh --ops=matinv_batched --run
```

## 文件结构

```
blas/matinv_batched/
├── README.md
└── arch35/
    ├── matinv_batched_tiling_data.h    # TilingData 结构体定义
    ├── matinv_batched_host.cpp         # Host 侧入口 + 参数校验 + 五步编排
    ├── copy_atoa_kernel.cpp            # CopyKernel（Step 1/3 共用，SIMT）
    ├── copy_atoa_kernel.h              # copy_kernel_do 声明
    ├── init_ptr_array_kernel.cpp       # InitPtrArrayKernel（Step 0，SIMT）
    └── init_ptr_array_kernel.h         # init_ptr_array_kernel_do 声明
```

内部复用（不修改）：

- `blas/getrf_batched/arch35/` — Step 2 LU 分解 kernel
- `blas/getri_batched/arch35/` — Step 4 求逆 kernel
